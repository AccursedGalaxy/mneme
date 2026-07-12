// Package sqlite is the pure-Go (modernc.org/sqlite, no cgo) default Store for
// mneme. Embeddings are stored as little-endian float32 BLOBs and cosine
// similarity is computed in Go over the scope-filtered rows. This is O(n) per
// scope — correct and simple, fine for thousands of facts. When that hurts,
// swap in a sqlite-vec or pgvector Store; the pipeline does not change.
package sqlite

import (
	"context"
	"database/sql"
	"encoding/binary"
	"fmt"
	"math"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"

	"github.com/AccursedGalaxy/mneme/store"
	"github.com/AccursedGalaxy/mneme/types"

	_ "modernc.org/sqlite"
)

const schema = `
CREATE TABLE IF NOT EXISTS facts (
  id          TEXT PRIMARY KEY,
  text        TEXT NOT NULL,
  hash        TEXT NOT NULL,
  embedding   BLOB NOT NULL,
  user_id     TEXT NOT NULL DEFAULT '',
  agent_id    TEXT NOT NULL DEFAULT '',
  run_id      TEXT NOT NULL DEFAULT '',
  created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_facts_scope ON facts(user_id, agent_id, run_id);
CREATE INDEX IF NOT EXISTS idx_facts_hash  ON facts(user_id, agent_id, run_id, hash);

CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
`

// meta keys recording the embedder identity (see store.EmbedderInfo). The model
// name and dimension are stored as separate rows so either can be absent on
// stores written by older versions without a migration.
const (
	metaEmbedderModel = "embedder_model"
	metaEmbedderDim   = "embedder_dim"
)

// Store is a SQLite-backed store.Store.
type Store struct {
	db *sql.DB
	// anchor pins one connection open for the store's lifetime when the
	// database is in-memory. Without it, database/sql may discard the pool's
	// connection (bad-conn error, context cancelled mid-query) and dial a
	// replacement — and for an in-memory database a replacement connection is a
	// brand-new empty database: total silent data loss. The anchor plus
	// cache=shared keeps the database alive across pool churn.
	anchor *sql.Conn
}

var _ store.Store = (*Store)(nil)

// memSeq disambiguates concurrent ":memory:" opens: each Open must get its own
// private database, so each gets a distinct shared-cache name.
var memSeq atomic.Uint64

// Open opens (creating if needed) a SQLite database at path and ensures the
// schema exists. Use ":memory:" for an ephemeral store. The returned *Store is
// safe for concurrent use by multiple goroutines.
//
// File-backed databases run in WAL mode (PRAGMA journal_mode=WAL) so reads run
// concurrently with a single writer — the read-heavy access pattern of an agent
// hitting memory every turn. busy_timeout lets a contended writer wait rather
// than fail immediately with "database is locked".
//
// ":memory:" is rewritten to a uniquely-named shared-cache in-memory database
// and one connection is held open for the store's lifetime, so the data
// survives connection-pool churn (see Store.anchor). A caller-supplied
// pre-formatted in-memory DSN (a "file:...mode=memory" string) is passed
// through unchanged and pinned to a single pooled connection instead — note
// that such a store does NOT survive that connection being discarded (e.g.
// after a context cancelled mid-query); prefer plain ":memory:".
func Open(path string) (*Store, error) {
	mem := isMemoryPath(path)
	private := path == ":memory:"
	if private {
		// A unique shared-cache name gives this Open its own database that all
		// pool connections see, instead of one private database per connection.
		path = fmt.Sprintf("file:mneme-mem-%d?mode=memory&cache=shared", memSeq.Add(1))
	}
	db, err := sql.Open("sqlite", dsn(path, mem))
	if err != nil {
		return nil, fmt.Errorf("open sqlite %q: %w", path, err)
	}
	s := &Store{db: db}
	if private {
		conn, err := db.Conn(context.Background())
		if err != nil {
			db.Close()
			return nil, fmt.Errorf("pin in-memory connection: %w", err)
		}
		s.anchor = conn
	} else if mem {
		db.SetMaxOpenConns(1)
	}
	if _, err := db.Exec(schema); err != nil {
		s.Close()
		return nil, fmt.Errorf("init schema: %w", err)
	}
	return s, nil
}

// isMemoryPath reports whether path names an in-memory database, which SQLite
// scopes to a single connection unless a shared cache is requested explicitly.
func isMemoryPath(path string) bool {
	return path == ":memory:" || strings.Contains(path, ":memory:") || strings.Contains(path, "mode=memory")
}

// dsn builds the modernc.org/sqlite connection string. File databases get WAL
// and busy_timeout pragmas applied per-connection (so every pooled connection
// inherits them); in-memory and pre-formatted DSNs are passed through unchanged.
func dsn(path string, mem bool) string {
	if mem || strings.HasPrefix(path, "file:") {
		return path
	}
	q := url.Values{}
	q.Add("_pragma", "journal_mode(WAL)")
	q.Add("_pragma", "busy_timeout(5000)")
	return "file:" + path + "?" + q.Encode()
}

// Insert persists a batch of records in one transaction. A record whose
// (scope, hash) already exists is silently skipped: the guard runs inside the
// write transaction, so two concurrent Adds that both extracted the same fact
// cannot both insert it — the pipeline's read-then-dedup pass is advisory and
// this is the authoritative backstop. (An Update writing a duplicate hash is
// deliberately NOT constrained — overwriting a fact in place is consolidation's
// documented behavior even when the new text matches another fact.)
//
// A zero-length embedding is rejected: it would be stored but score 0 against
// every query — a fact that silently never surfaces — so it fails loudly here.
func (s *Store) Insert(ctx context.Context, recs []types.Record) error {
	if len(recs) == 0 {
		return nil
	}
	for _, r := range recs {
		if len(r.Embedding) == 0 {
			return fmt.Errorf("insert %s: zero-length embedding", r.ID)
		}
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	stmt, err := tx.PrepareContext(ctx, `INSERT INTO facts
		(id, text, hash, embedding, user_id, agent_id, run_id, created_at)
		SELECT ?, ?, ?, ?, ?, ?, ?, ?
		WHERE NOT EXISTS (SELECT 1 FROM facts
			WHERE user_id = ? AND agent_id = ? AND run_id = ? AND hash = ?)`)
	if err != nil {
		return err
	}
	defer stmt.Close()
	for _, r := range recs {
		_, err := stmt.ExecContext(ctx,
			r.ID, r.Text, r.Hash, encodeVec(r.Embedding),
			r.Scope.UserID, r.Scope.AgentID, r.Scope.RunID,
			r.CreatedAt.UTC().Format(timeLayout),
			r.Scope.UserID, r.Scope.AgentID, r.Scope.RunID, r.Hash,
		)
		if err != nil {
			return fmt.Errorf("insert %s: %w", r.ID, err)
		}
	}
	return tx.Commit()
}

// Update replaces a record's text, hash and embedding by id, leaving its scope
// and created_at untouched (created_at is the ingestion time, which an update
// does not change). Updating a missing id is a no-op, not an error; the returned
// bool reports whether a row actually matched.
func (s *Store) Update(ctx context.Context, rec types.Record) (bool, error) {
	res, err := s.db.ExecContext(ctx,
		`UPDATE facts SET text = ?, hash = ?, embedding = ? WHERE id = ?`,
		rec.Text, rec.Hash, encodeVec(rec.Embedding), rec.ID)
	if err != nil {
		return false, fmt.Errorf("update %s: %w", rec.ID, err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return false, fmt.Errorf("update %s: rows affected: %w", rec.ID, err)
	}
	return n > 0, nil
}

func (s *Store) Search(ctx context.Context, scope types.Scope, vec []float32, k int) ([]types.Hit, error) {
	if k <= 0 {
		return nil, nil
	}
	rows, err := s.db.QueryContext(ctx,
		`SELECT id, text, hash, embedding, user_id, agent_id, run_id, created_at
		 FROM facts WHERE user_id = ? AND agent_id = ? AND run_id = ?`,
		scope.UserID, scope.AgentID, scope.RunID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var hits []types.Hit
	for rows.Next() {
		rec, err := scanRecord(rows)
		if err != nil {
			return nil, err
		}
		hits = append(hits, types.Hit{Record: rec, Score: store.Cosine(vec, rec.Embedding)})
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	// Sort by score desc; break ties by created_at then id for determinism.
	sort.Slice(hits, func(i, j int) bool {
		if hits[i].Score != hits[j].Score {
			return hits[i].Score > hits[j].Score
		}
		if !hits[i].CreatedAt.Equal(hits[j].CreatedAt) {
			return hits[i].CreatedAt.After(hits[j].CreatedAt)
		}
		return hits[i].ID < hits[j].ID
	})
	if len(hits) > k {
		hits = hits[:k]
	}
	return hits, nil
}

func (s *Store) Get(ctx context.Context, id string) (types.Record, error) {
	row := s.db.QueryRowContext(ctx,
		`SELECT id, text, hash, embedding, user_id, agent_id, run_id, created_at
		 FROM facts WHERE id = ?`, id)
	rec, err := scanRecord(row)
	if err == sql.ErrNoRows {
		return types.Record{}, fmt.Errorf("fact %q: %w", id, store.ErrNotFound)
	}
	return rec, err
}

func (s *Store) Delete(ctx context.Context, id string) error {
	_, err := s.db.ExecContext(ctx, `DELETE FROM facts WHERE id = ?`, id)
	return err
}

func (s *Store) ExistingHashes(ctx context.Context, scope types.Scope) (map[string]struct{}, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT hash FROM facts WHERE user_id = ? AND agent_id = ? AND run_id = ?`,
		scope.UserID, scope.AgentID, scope.RunID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make(map[string]struct{})
	for rows.Next() {
		var h string
		if err := rows.Scan(&h); err != nil {
			return nil, err
		}
		out[h] = struct{}{}
	}
	return out, rows.Err()
}

// EmbedderMeta reads the recorded embedder identity. ok is false when nothing
// has been recorded yet (presence is keyed on the dimension row, always written
// by SetEmbedderMeta; the model row may legitimately be empty for an unnamed
// embedder).
func (s *Store) EmbedderMeta(ctx context.Context) (store.EmbedderInfo, bool, error) {
	rows, err := s.db.QueryContext(ctx, `SELECT key, value FROM meta WHERE key IN (?, ?)`,
		metaEmbedderModel, metaEmbedderDim)
	if err != nil {
		return store.EmbedderInfo{}, false, err
	}
	defer rows.Close()
	var (
		info   store.EmbedderInfo
		hasDim bool
	)
	for rows.Next() {
		var key, value string
		if err := rows.Scan(&key, &value); err != nil {
			return store.EmbedderInfo{}, false, err
		}
		switch key {
		case metaEmbedderModel:
			info.Model = value
		case metaEmbedderDim:
			d, err := strconv.Atoi(value)
			if err != nil {
				return store.EmbedderInfo{}, false, fmt.Errorf("parse embedder dim %q: %w", value, err)
			}
			info.Dim = d
			hasDim = true
		}
	}
	if err := rows.Err(); err != nil {
		return store.EmbedderInfo{}, false, err
	}
	return info, hasDim, nil
}

// SetEmbedderMeta upserts the embedder identity rows.
func (s *Store) SetEmbedderMeta(ctx context.Context, info store.EmbedderInfo) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	const upsert = `INSERT INTO meta (key, value) VALUES (?, ?)
		ON CONFLICT(key) DO UPDATE SET value = excluded.value`
	if _, err := tx.ExecContext(ctx, upsert, metaEmbedderModel, info.Model); err != nil {
		return err
	}
	if _, err := tx.ExecContext(ctx, upsert, metaEmbedderDim, strconv.Itoa(info.Dim)); err != nil {
		return err
	}
	return tx.Commit()
}

func (s *Store) Close() error {
	if s.anchor != nil {
		s.anchor.Close() // releases the pinned in-memory connection
	}
	return s.db.Close()
}

const timeLayout = "2006-01-02T15:04:05.999999999Z07:00" // RFC3339Nano

// scanner is satisfied by both *sql.Row and *sql.Rows.
type scanner interface {
	Scan(dest ...any) error
}

func scanRecord(sc scanner) (types.Record, error) {
	var (
		rec     types.Record
		blob    []byte
		created string
	)
	if err := sc.Scan(&rec.ID, &rec.Text, &rec.Hash, &blob,
		&rec.Scope.UserID, &rec.Scope.AgentID, &rec.Scope.RunID, &created); err != nil {
		return types.Record{}, err
	}
	vec, err := decodeVec(blob)
	if err != nil {
		return types.Record{}, fmt.Errorf("record %s: %w", rec.ID, err)
	}
	rec.Embedding = vec
	t, err := parseTime(created)
	if err != nil {
		return types.Record{}, fmt.Errorf("parse created_at %q: %w", created, err)
	}
	rec.CreatedAt = t
	return rec, nil
}

func encodeVec(v []float32) []byte {
	b := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(f))
	}
	return b
}

// decodeVec is the inverse of encodeVec. A blob whose length is not a multiple
// of 4 is corrupt (truncated or foreign data) and surfaces as an error rather
// than silently decoding to a shorter vector that scores 0 forever.
func decodeVec(b []byte) ([]float32, error) {
	if len(b)%4 != 0 {
		return nil, fmt.Errorf("corrupt embedding blob: %d bytes is not a multiple of 4", len(b))
	}
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return v, nil
}
