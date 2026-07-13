package sqlite

import (
	"context"
	"database/sql"
	"errors"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/AccursedGalaxy/mneme/store"
	"github.com/AccursedGalaxy/mneme/types"
)

func newStore(t *testing.T) *Store {
	t.Helper()
	path := filepath.Join(t.TempDir(), "test.db")
	s, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

func rec(id, text string, vec []float32, sc types.Scope) types.Record {
	return types.Record{
		ID:        id,
		Text:      text,
		Hash:      "h-" + id,
		Embedding: vec,
		Scope:     sc,
		CreatedAt: time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC),
	}
}

func TestInsertGetDelete(t *testing.T) {
	s := newStore(t)
	ctx := context.Background()
	sc := types.Scope{UserID: "alice"}
	in := rec("1", "Alice drives a Ferrari", []float32{0.1, 0.2, 0.3}, sc)

	if err := s.Insert(ctx, []types.Record{in}); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	got, err := s.Get(ctx, "1")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got.Text != in.Text || got.Hash != in.Hash {
		t.Errorf("Get mismatch: %+v", got)
	}
	if len(got.Embedding) != 3 || got.Embedding[1] != 0.2 {
		t.Errorf("embedding round-trip failed: %v", got.Embedding)
	}
	if !got.CreatedAt.Equal(in.CreatedAt) {
		t.Errorf("created_at round-trip: got %v want %v", got.CreatedAt, in.CreatedAt)
	}
	if got.Scope != sc {
		t.Errorf("scope round-trip: got %+v", got.Scope)
	}

	if err := s.Delete(ctx, "1"); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if _, err := s.Get(ctx, "1"); !errors.Is(err, store.ErrNotFound) {
		t.Errorf("Get after delete: want ErrNotFound, got %v", err)
	}
	// Deleting a missing id is not an error.
	if err := s.Delete(ctx, "nope"); err != nil {
		t.Errorf("Delete missing: %v", err)
	}
}

func TestUpdate(t *testing.T) {
	s := newStore(t)
	ctx := context.Background()
	sc := types.Scope{UserID: "alice"}
	orig := rec("1", "Alice lives in Seattle", []float32{1, 0, 0}, sc)
	must(t, s.Insert(ctx, []types.Record{orig}))

	// Update text/hash/embedding; scope + created_at must survive untouched.
	upd := types.Record{ID: "1", Text: "Alice lives in Austin", Hash: "h-new", Embedding: []float32{0, 1, 0}}
	updated, err := s.Update(ctx, upd)
	if err != nil {
		t.Fatalf("Update: %v", err)
	}
	if !updated {
		t.Errorf("Update of an existing id should report a match")
	}
	got, err := s.Get(ctx, "1")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got.Text != "Alice lives in Austin" || got.Hash != "h-new" {
		t.Errorf("text/hash not updated: %+v", got)
	}
	if len(got.Embedding) != 3 || got.Embedding[1] != 1 {
		t.Errorf("embedding not updated: %v", got.Embedding)
	}
	if got.Scope != sc {
		t.Errorf("scope should be preserved by Update: %+v", got.Scope)
	}
	if !got.CreatedAt.Equal(orig.CreatedAt) {
		t.Errorf("created_at should be preserved by Update: got %v want %v", got.CreatedAt, orig.CreatedAt)
	}

	// Updating a missing id is a no-op, not an error, and reports no match.
	updated, err = s.Update(ctx, types.Record{ID: "nope", Text: "x", Hash: "h", Embedding: []float32{1}})
	if err != nil {
		t.Errorf("Update missing id should be a no-op: %v", err)
	}
	if updated {
		t.Errorf("Update of a missing id must report no match")
	}
	if _, err := s.Get(ctx, "nope"); !errors.Is(err, store.ErrNotFound) {
		t.Errorf("missing-id update must not create a row, got %v", err)
	}
}

func TestSearchNearestNeighbourOrder(t *testing.T) {
	s := newStore(t)
	ctx := context.Background()
	sc := types.Scope{UserID: "alice"}

	// Query will be {1,0}. Closeness to it: a > b > c.
	must(t, s.Insert(ctx, []types.Record{
		rec("a", "closest", []float32{1, 0.1}, sc),
		rec("b", "middle", []float32{1, 1}, sc),
		rec("c", "farthest", []float32{-1, 0}, sc),
	}))

	hits, err := s.Search(ctx, sc, []float32{1, 0}, 3)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) != 3 {
		t.Fatalf("want 3 hits, got %d", len(hits))
	}
	order := []string{hits[0].ID, hits[1].ID, hits[2].ID}
	want := []string{"a", "b", "c"}
	for i := range want {
		if order[i] != want[i] {
			t.Fatalf("nearest-neighbour order = %v, want %v", order, want)
		}
	}
	// Scores descending.
	if !(hits[0].Score >= hits[1].Score && hits[1].Score >= hits[2].Score) {
		t.Errorf("scores not descending: %v %v %v", hits[0].Score, hits[1].Score, hits[2].Score)
	}
}

func TestSearchTopK(t *testing.T) {
	s := newStore(t)
	ctx := context.Background()
	sc := types.Scope{UserID: "alice"}
	must(t, s.Insert(ctx, []types.Record{
		rec("a", "a", []float32{1, 0}, sc),
		rec("b", "b", []float32{0.9, 0.1}, sc),
		rec("c", "c", []float32{0.1, 1}, sc),
	}))
	hits, err := s.Search(ctx, sc, []float32{1, 0}, 2)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) != 2 {
		t.Fatalf("k=2 should cap at 2, got %d", len(hits))
	}
}

func TestScopeIsolation(t *testing.T) {
	s := newStore(t)
	ctx := context.Background()
	alice := types.Scope{UserID: "alice"}
	bob := types.Scope{UserID: "bob"}

	must(t, s.Insert(ctx, []types.Record{
		rec("a1", "alice fact", []float32{1, 0}, alice),
		rec("b1", "bob fact", []float32{1, 0}, bob),
	}))

	hits, err := s.Search(ctx, alice, []float32{1, 0}, 10)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) != 1 || hits[0].ID != "a1" {
		t.Fatalf("scope isolation broken: %+v", hits)
	}

	hashes, err := s.ExistingHashes(ctx, bob)
	if err != nil {
		t.Fatalf("ExistingHashes: %v", err)
	}
	if len(hashes) != 1 {
		t.Fatalf("want 1 hash for bob, got %d", len(hashes))
	}
	if _, ok := hashes["h-b1"]; !ok {
		t.Errorf("bob's hash missing: %v", hashes)
	}
}

func TestExistingHashes(t *testing.T) {
	s := newStore(t)
	ctx := context.Background()
	sc := types.Scope{UserID: "alice"}
	must(t, s.Insert(ctx, []types.Record{
		rec("1", "one", []float32{1, 0}, sc),
		rec("2", "two", []float32{0, 1}, sc),
	}))
	hashes, err := s.ExistingHashes(ctx, sc)
	if err != nil {
		t.Fatalf("ExistingHashes: %v", err)
	}
	for _, want := range []string{"h-1", "h-2"} {
		if _, ok := hashes[want]; !ok {
			t.Errorf("missing hash %q in %v", want, hashes)
		}
	}
}

func TestPersistenceAcrossReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "persist.db")
	sc := types.Scope{UserID: "alice"}

	s1, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	must(t, s1.Insert(ctx, []types.Record{rec("1", "persisted", []float32{1, 2, 3}, sc)}))
	s1.Close()

	s2, err := Open(path)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer s2.Close()
	got, err := s2.Get(ctx, "1")
	if err != nil {
		t.Fatalf("Get after reopen: %v", err)
	}
	if got.Text != "persisted" {
		t.Errorf("lost data across reopen: %+v", got)
	}
}

// TestWALModeEnabled pins the access mode of a file-backed store: WAL lets
// readers run concurrently with a single writer, and busy_timeout keeps a
// contended writer waiting instead of failing immediately.
func TestWALModeEnabled(t *testing.T) {
	s := newStore(t) // newStore uses a real temp-file path, not :memory:
	ctx := context.Background()

	var jm string
	if err := s.db.QueryRowContext(ctx, `PRAGMA journal_mode`).Scan(&jm); err != nil {
		t.Fatalf("query journal_mode: %v", err)
	}
	if jm != "wal" {
		t.Errorf("journal_mode = %q, want wal", jm)
	}

	var bt int
	if err := s.db.QueryRowContext(ctx, `PRAGMA busy_timeout`).Scan(&bt); err != nil {
		t.Fatalf("query busy_timeout: %v", err)
	}
	if bt != 5000 {
		t.Errorf("busy_timeout = %d, want 5000", bt)
	}
}

// TestConcurrentReadsAndWrites is the payoff of WAL: many readers proceed while
// a writer commits, with no "database is locked". Before WAL this Store pinned
// itself to a single connection to dodge that error; now the pool is open and
// busy_timeout absorbs writer contention. The race detector guards the test.
func TestConcurrentReadsAndWrites(t *testing.T) {
	s := newStore(t)
	ctx := context.Background()
	sc := types.Scope{UserID: "alice"}
	must(t, s.Insert(ctx, []types.Record{rec("seed", "seed fact", []float32{1, 0}, sc)}))

	var wg sync.WaitGroup
	errc := make(chan error, 32)

	// Readers hammer Search concurrently.
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 25; j++ {
				if _, err := s.Search(ctx, sc, []float32{1, 0}, 5); err != nil {
					errc <- err
					return
				}
			}
		}()
	}
	// Writers insert distinct facts concurrently with the readers.
	for i := 0; i < 4; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 25; j++ {
				id := strconv.Itoa(i*100 + j)
				if err := s.Insert(ctx, []types.Record{rec(id, "f"+id, []float32{0, 1}, sc)}); err != nil {
					errc <- err
					return
				}
			}
		}()
	}

	wg.Wait()
	close(errc)
	for err := range errc {
		t.Fatalf("concurrent access error: %v", err)
	}
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

func TestEmbedderMetaRoundTrip(t *testing.T) {
	s := newStore(t)
	ctx := context.Background()

	// A fresh store records no identity.
	if _, ok, err := s.EmbedderMeta(ctx); err != nil || ok {
		t.Fatalf("EmbedderMeta on fresh store: ok=%v err=%v, want ok=false", ok, err)
	}

	want := store.EmbedderInfo{Model: "text-embedding-3-small", Dim: 1536}
	if err := s.SetEmbedderMeta(ctx, want); err != nil {
		t.Fatalf("SetEmbedderMeta: %v", err)
	}
	got, ok, err := s.EmbedderMeta(ctx)
	if err != nil || !ok {
		t.Fatalf("EmbedderMeta after set: ok=%v err=%v", ok, err)
	}
	if got != want {
		t.Errorf("EmbedderMeta = %+v, want %+v", got, want)
	}

	// SetEmbedderMeta overwrites, and an unnamed embedder (empty model) still
	// records — presence is keyed on the dimension row, not the model.
	next := store.EmbedderInfo{Model: "", Dim: 64}
	if err := s.SetEmbedderMeta(ctx, next); err != nil {
		t.Fatalf("SetEmbedderMeta overwrite: %v", err)
	}
	got, ok, err = s.EmbedderMeta(ctx)
	if err != nil || !ok {
		t.Fatalf("EmbedderMeta after overwrite: ok=%v err=%v", ok, err)
	}
	if got != next {
		t.Errorf("EmbedderMeta = %+v, want %+v", got, next)
	}
}

func TestInsertBackstopsConcurrentHashRace(t *testing.T) {
	// The pipeline's read-then-dedup is advisory; the store is the
	// authoritative backstop. Two batches carrying the same (scope, hash) —
	// what two racing Adds produce — must yield exactly one row.
	s := newStore(t)
	ctx := context.Background()
	scope := types.Scope{UserID: "u"}
	rec := func(id string) types.Record {
		return types.Record{
			ID: id, Text: "same fact", Hash: "h1",
			Embedding: []float32{1, 2}, Scope: scope, CreatedAt: time.Now(),
		}
	}
	if err := s.Insert(ctx, []types.Record{rec("id-a")}); err != nil {
		t.Fatalf("first insert: %v", err)
	}
	if err := s.Insert(ctx, []types.Record{rec("id-b")}); err != nil {
		t.Fatalf("duplicate insert must be a silent no-op, got %v", err)
	}
	hits, err := s.Search(ctx, scope, []float32{1, 2}, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 {
		t.Errorf("duplicate (scope, hash) must not create a second row: got %d rows", len(hits))
	}
	// A different scope with the same hash is a different fact-space and must insert.
	other := rec("id-c")
	other.Scope = types.Scope{UserID: "v"}
	if err := s.Insert(ctx, []types.Record{other}); err != nil {
		t.Fatalf("same hash in another scope: %v", err)
	}
	hits, err = s.Search(ctx, other.Scope, []float32{1, 2}, 10)
	if err != nil || len(hits) != 1 {
		t.Errorf("other scope should hold its own copy: %d rows, err=%v", len(hits), err)
	}
}

func TestInsertRejectsEmptyEmbedding(t *testing.T) {
	s := newStore(t)
	err := s.Insert(context.Background(), []types.Record{{
		ID: "x", Text: "t", Hash: "h", Scope: types.Scope{}, CreatedAt: time.Now(),
	}})
	if err == nil || !strings.Contains(err.Error(), "zero-length embedding") {
		t.Errorf("zero-length embedding must be rejected, got %v", err)
	}
}

func TestUpdateMayDuplicateHash(t *testing.T) {
	// Consolidation's documented contract: an UPDATE overwrites in place even
	// when its new text (hash) matches another stored fact. The insert-time
	// dedup backstop must not constrain updates.
	s := newStore(t)
	ctx := context.Background()
	scope := types.Scope{UserID: "u"}
	recs := []types.Record{
		{ID: "a", Text: "fact one", Hash: "h-one", Embedding: []float32{1}, Scope: scope, CreatedAt: time.Now()},
		{ID: "b", Text: "fact two", Hash: "h-two", Embedding: []float32{1}, Scope: scope, CreatedAt: time.Now()},
	}
	if err := s.Insert(ctx, recs); err != nil {
		t.Fatal(err)
	}
	ok, err := s.Update(ctx, types.Record{ID: "b", Text: "fact one", Hash: "h-one", Embedding: []float32{1}})
	if err != nil || !ok {
		t.Fatalf("update to a duplicate hash must succeed: ok=%v err=%v", ok, err)
	}
}

func TestMemoryStoreSurvivesConnectionChurn(t *testing.T) {
	// Every pool connection must see the same in-memory database, and the data
	// must survive the pool discarding and re-dialing connections — the failure
	// mode where a context cancelled mid-query silently wiped a ":memory:"
	// store.
	s, err := Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	ctx := context.Background()
	scope := types.Scope{UserID: "u"}
	if err := s.Insert(ctx, []types.Record{{
		ID: "a", Text: "keep me", Hash: "h", Embedding: []float32{1}, Scope: scope, CreatedAt: time.Now(),
	}}); err != nil {
		t.Fatal(err)
	}

	// Force pool churn: forbid idle connections so each query dials a fresh
	// one, then query several times. Pre-fix, each fresh connection was a new
	// empty database.
	s.db.SetMaxIdleConns(0)
	for i := 0; i < 3; i++ {
		hits, err := s.Search(ctx, scope, []float32{1}, 10)
		if err != nil {
			t.Fatalf("search %d: %v", i, err)
		}
		if len(hits) != 1 {
			t.Fatalf("in-memory data lost after connection churn (query %d): %d rows", i, len(hits))
		}
	}
}

func TestMemoryStoresAreIsolated(t *testing.T) {
	// Two ":memory:" opens must be two separate databases.
	a, err := Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	b, err := Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	ctx := context.Background()
	scope := types.Scope{UserID: "u"}
	if err := a.Insert(ctx, []types.Record{{
		ID: "a", Text: "in store a", Hash: "h", Embedding: []float32{1}, Scope: scope, CreatedAt: time.Now(),
	}}); err != nil {
		t.Fatal(err)
	}
	hits, err := b.Search(ctx, scope, []float32{1}, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 0 {
		t.Errorf("store b sees store a's rows: %d", len(hits))
	}
}

func TestCorruptEmbeddingBlobSurfacesError(t *testing.T) {
	// A blob whose length is not a multiple of 4 is corrupt; reads must error
	// loudly instead of silently decoding a truncated vector that scores 0.
	s := newStore(t)
	ctx := context.Background()
	scope := types.Scope{UserID: "u"}
	if err := s.Insert(ctx, []types.Record{{
		ID: "a", Text: "t", Hash: "h", Embedding: []float32{1}, Scope: scope, CreatedAt: time.Now(),
	}}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.db.Exec(`UPDATE facts SET embedding = ? WHERE id = 'a'`, []byte{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Search(ctx, scope, []float32{1}, 10); err == nil || !strings.Contains(err.Error(), "corrupt embedding") {
		t.Errorf("corrupt blob must surface an error, got %v", err)
	}
}

// A database written before observed_at existed must keep working: CREATE TABLE
// IF NOT EXISTS silently skips an existing table, so without the migration every
// query naming the new column fails with "no such column" and an upgrade bricks
// the caller's store.
func TestOpenMigratesPreObservedAtDatabase(t *testing.T) {
	path := filepath.Join(t.TempDir(), "old.db")

	// Build a store with the old schema, by hand, and put a row in it.
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	_, err = db.Exec(`CREATE TABLE facts (
	  id TEXT PRIMARY KEY, text TEXT NOT NULL, hash TEXT NOT NULL,
	  embedding BLOB NOT NULL, user_id TEXT NOT NULL DEFAULT '',
	  agent_id TEXT NOT NULL DEFAULT '', run_id TEXT NOT NULL DEFAULT '',
	  created_at TEXT NOT NULL);
	INSERT INTO facts (id, text, hash, embedding, user_id, created_at)
	  VALUES ('old-1', 'a fact from before', 'h1', x'0000803f', 'u', '2026-01-01T00:00:00Z');`)
	if err != nil {
		t.Fatal(err)
	}
	db.Close()

	st, err := Open(path)
	if err != nil {
		t.Fatalf("opening a pre-migration database must succeed: %v", err)
	}
	defer st.Close()

	got, err := st.Get(context.Background(), "old-1")
	if err != nil {
		t.Fatalf("reading a pre-migration row must succeed: %v", err)
	}
	if got.Text != "a fact from before" {
		t.Errorf("text = %q, want the original", got.Text)
	}
	// The row predates the column: its source timestamp is genuinely unknown.
	if !got.ObservedAt.IsZero() {
		t.Errorf("a migrated row's ObservedAt must be zero (unknown), got %v", got.ObservedAt)
	}

	// And the migrated store must still accept new writes carrying the column.
	at := time.Date(2023, 5, 8, 13, 56, 0, 0, time.UTC)
	if err := st.Insert(context.Background(), []types.Record{{
		ID: "new-1", Text: "a fact from after", Hash: "h2",
		Embedding: []float32{1}, Scope: types.Scope{UserID: "u"},
		CreatedAt: time.Now(), ObservedAt: at,
	}}); err != nil {
		t.Fatal(err)
	}
	back, err := st.Get(context.Background(), "new-1")
	if err != nil {
		t.Fatal(err)
	}
	if !back.ObservedAt.Equal(at) {
		t.Errorf("observed_at round-trip: got %v want %v", back.ObservedAt, at)
	}
}

// Open runs the migration on every call, so it must be idempotent.
func TestMigrateIsIdempotent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "x.db")
	for i := 0; i < 3; i++ {
		st, err := Open(path)
		if err != nil {
			t.Fatalf("open %d: %v", i, err)
		}
		st.Close()
	}
}

// Update must carry observed_at. It did not, and the omission silently discarded
// the ObservedAt the consolidation path had just computed: the row kept its old
// (usually empty) date, so the fact read back undated and the whole temporal
// grounding mechanism was a no-op for every consolidated fact.
func TestUpdateWritesObservedAt(t *testing.T) {
	st := newStore(t)
	ctx := context.Background()

	if err := st.Insert(ctx, []types.Record{{
		ID: "f1", Text: "Caroline works at Acme", Hash: "h1",
		Embedding: []float32{1}, Scope: types.Scope{UserID: "u"},
		CreatedAt: time.Now(),
	}}); err != nil {
		t.Fatal(err)
	}

	said := time.Date(2023, 6, 1, 9, 0, 0, 0, time.UTC)
	updated, err := st.Update(ctx, types.Record{
		ID: "f1", Text: "Caroline works at Globex", Hash: "h2",
		Embedding: []float32{1}, ObservedAt: said,
	})
	if err != nil || !updated {
		t.Fatalf("update: %v, updated=%v", err, updated)
	}

	got, err := st.Get(ctx, "f1")
	if err != nil {
		t.Fatal(err)
	}
	if !got.ObservedAt.Equal(said) {
		t.Errorf("ObservedAt = %v, want the updating conversation's time %v", got.ObservedAt, said)
	}
}

// An update from an undated conversation must keep the date the fact already
// had. Blanking it would turn the silent-drop bug into a silent-erase bug: a
// previously answerable "when did X happen" would stop resolving.
func TestUpdateWithZeroObservedAtPreservesStoredDate(t *testing.T) {
	st := newStore(t)
	ctx := context.Background()

	said := time.Date(2023, 5, 8, 13, 56, 0, 0, time.UTC)
	if err := st.Insert(ctx, []types.Record{{
		ID: "f1", Text: "Caroline attended a support group", Hash: "h1",
		Embedding: []float32{1}, Scope: types.Scope{UserID: "u"},
		CreatedAt: time.Now(), ObservedAt: said,
	}}); err != nil {
		t.Fatal(err)
	}

	// A reconciling conversation that carried no timestamp: ObservedAt is zero.
	if _, err := st.Update(ctx, types.Record{
		ID: "f1", Text: "Caroline attended an LGBTQ support group", Hash: "h2",
		Embedding: []float32{1},
	}); err != nil {
		t.Fatal(err)
	}

	got, err := st.Get(ctx, "f1")
	if err != nil {
		t.Fatal(err)
	}
	if !got.ObservedAt.Equal(said) {
		t.Errorf("an undated update must preserve the stored date, got %v want %v", got.ObservedAt, said)
	}
	if got.Text != "Caroline attended an LGBTQ support group" {
		t.Errorf("the text must still be updated, got %q", got.Text)
	}
}
