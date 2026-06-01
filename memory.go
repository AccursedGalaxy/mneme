package mneme

import (
	"context"
	"crypto/rand"
	"fmt"
	"time"

	"github.com/AccursedGalaxy/mneme/provider"
	"github.com/AccursedGalaxy/mneme/store"
	"github.com/AccursedGalaxy/mneme/types"
)

// Public value types are defined once in the leaf package types and re-exported
// here as aliases, so consumers use mneme.Message / mneme.Scope / mneme.Fact
// while the store package can share the same types without an import cycle.
type (
	Message = types.Message
	Scope   = types.Scope
	Fact    = types.Fact
)

// DefaultExtractionTopK is how many existing memories are shown to the extractor
// for dedup/linking when WithExtractionTopK is not set.
const DefaultExtractionTopK = 10

// Memory is the top-level handle an agent holds. It is safe for concurrent use
// to the extent its underlying store, LLM and embedder are.
type Memory interface {
	// Add ingests messages, extracts durable facts, dedups, and stores them.
	// It returns the facts that were newly written (after dedup).
	Add(ctx context.Context, msgs []Message, scope Scope) ([]Fact, error)

	// Search returns the top-k facts most relevant to query within scope,
	// with Score populated, highest first.
	Search(ctx context.Context, query string, scope Scope, k int) ([]Fact, error)

	// Get returns one fact by id.
	Get(ctx context.Context, id string) (Fact, error)

	// Delete removes one fact by id.
	Delete(ctx context.Context, id string) error

	// Close releases resources (the DB handle).
	Close() error
}

// memory is the concrete Memory implementation wiring a store, an LLM and an
// embedder through the additive pipeline (see pipeline.go).
type memory struct {
	store          store.Store
	llm            provider.LLM
	embedder       provider.Embedder
	extractionTopK int
	promptVersion  string
	clock          func() time.Time
}

var _ Memory = (*memory)(nil)

// Option configures New. Options are applied in order; later ones win.
type Option func(*memory)

// WithStore sets the persistence backend. Without it, New builds a SQLite store
// at MNEME_DB_PATH (default ./mneme.db).
func WithStore(s store.Store) Option { return func(m *memory) { m.store = s } }

// WithLLM sets the extraction LLM. Without it, New builds an OpenAI-compatible
// client from MNEME_LLM_* env vars.
func WithLLM(l provider.LLM) Option { return func(m *memory) { m.llm = l } }

// WithEmbedder sets the embedder. Without it, New builds an OpenAI-compatible
// client from MNEME_EMBED_* env vars.
func WithEmbedder(e provider.Embedder) Option { return func(m *memory) { m.embedder = e } }

// WithExtractionTopK sets how many existing memories are shown to the extractor.
func WithExtractionTopK(n int) Option {
	return func(m *memory) {
		if n >= 0 {
			m.extractionTopK = n
		}
	}
}

// WithPromptVersion selects the extraction prompt version (see PromptVersions).
func WithPromptVersion(v string) Option { return func(m *memory) { m.promptVersion = v } }

// New constructs a Memory. Any of the store, LLM and embedder not supplied via
// options are built from the MNEME_* environment (see PLAN.md §8). It returns an
// error if a required provider can be neither supplied nor built.
func New(opts ...Option) (Memory, error) {
	m := &memory{
		extractionTopK: DefaultExtractionTopK,
		promptVersion:  DefaultPromptVersion,
		clock:          time.Now,
	}
	for _, opt := range opts {
		opt(m)
	}

	if m.llm == nil {
		llm, err := llmFromEnv()
		if err != nil {
			return nil, err
		}
		m.llm = llm
	}
	if m.embedder == nil {
		emb, err := embedderFromEnv()
		if err != nil {
			return nil, err
		}
		m.embedder = emb
	}
	if m.store == nil {
		st, err := storeFromEnv()
		if err != nil {
			return nil, err
		}
		m.store = st
	}
	return m, nil
}

func (m *memory) Get(ctx context.Context, id string) (Fact, error) {
	rec, err := m.store.Get(ctx, id)
	if err != nil {
		return Fact{}, err
	}
	return recordToFact(rec, 0), nil
}

func (m *memory) Delete(ctx context.Context, id string) error {
	return m.store.Delete(ctx, id)
}

func (m *memory) Close() error { return m.store.Close() }

// recordToFact projects a stored Record into the caller-facing Fact (dropping
// the embedding), optionally attaching a search score.
func recordToFact(r types.Record, score float32) Fact {
	return Fact{
		ID:        r.ID,
		Text:      r.Text,
		Hash:      r.Hash,
		Score:     score,
		CreatedAt: r.CreatedAt,
		Scope:     r.Scope,
	}
}

// newUUID returns a random RFC-4122 v4 UUID string using crypto/rand, avoiding
// any third-party uuid dependency.
func newUUID() (string, error) {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	b[6] = (b[6] & 0x0f) | 0x40 // version 4
	b[8] = (b[8] & 0x3f) | 0x80 // variant 10
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16]), nil
}
