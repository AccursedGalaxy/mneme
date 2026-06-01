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

// Store persists facts and answers scoped vector searches.
type Store interface {
	// Insert writes records. It is the caller's job to have deduped first.
	Insert(ctx context.Context, recs []types.Record) error

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

	// Close releases resources (e.g. the DB handle).
	Close() error
}
