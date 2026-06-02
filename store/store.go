// Package store defines the persistence interface for mneme facts and ships a
// pure-Go SQLite implementation (store/sqlite). The interface is deliberately
// small so a future sqlite-vec or pgvector backend can drop in without touching
// the pipeline.
package store

import (
	"context"
	"errors"

	"github.com/AccursedGalaxy/mneme/types"
)

// ErrNotFound is returned by Get when no record has the requested id.
var ErrNotFound = errors.New("mneme: fact not found")

// EmbedderInfo identifies the embedding model whose vectors populate a store.
// mneme records it once, on the first insert, and checks it on New so that
// pointing a store at a different embedder is caught loudly instead of silently
// degrading search (cosine over vectors from two different models is noise).
type EmbedderInfo struct {
	Model string // model identifier; "" when the embedder is unnamed
	Dim   int    // vector dimension
}

// Store persists facts and answers scoped vector searches.
type Store interface {
	// Insert writes records. It is the caller's job to have deduped first.
	Insert(ctx context.Context, recs []types.Record) error

	// Update replaces an existing record's text, hash and embedding (matched by
	// id), leaving its scope and creation time intact. It is used by the
	// consolidation strategy's UPDATE op (mneme.WithStrategy(Consolidate)).
	// Updating a missing id is not an error; the returned bool reports whether a
	// row actually matched, letting callers tell a real write from a no-op (e.g.
	// a target concurrently deleted between retrieval and update).
	Update(ctx context.Context, rec types.Record) (bool, error)

	// Search returns up to k records in the given scope ranked by cosine
	// similarity to vec, highest first.
	Search(ctx context.Context, scope types.Scope, vec []float32, k int) ([]types.Hit, error)

	// Get returns one record by id.
	Get(ctx context.Context, id string) (types.Record, error)

	// Delete removes one record by id. Deleting a missing id is not an error.
	Delete(ctx context.Context, id string) error

	// ExistingHashes returns the set of fact hashes already stored in scope,
	// used by the pipeline to dedup before insert.
	ExistingHashes(ctx context.Context, scope types.Scope) (map[string]struct{}, error)

	// EmbedderMeta returns the embedder identity recorded for this store, with
	// ok=false when none has been recorded yet (a store with no facts). It lets
	// New detect an accidental embedder swap before search silently degrades.
	EmbedderMeta(ctx context.Context) (info EmbedderInfo, ok bool, err error)

	// SetEmbedderMeta records the store's embedder identity. The pipeline calls
	// it once, when the first fact is inserted; it overwrites any prior value.
	SetEmbedderMeta(ctx context.Context, info EmbedderInfo) error

	// Close releases resources (e.g. the DB handle).
	Close() error
}
