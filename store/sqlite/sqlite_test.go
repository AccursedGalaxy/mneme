package sqlite

import (
	"context"
	"errors"
	"path/filepath"
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
	if err := s.Update(ctx, upd); err != nil {
		t.Fatalf("Update: %v", err)
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

	// Updating a missing id is a no-op, not an error.
	if err := s.Update(ctx, types.Record{ID: "nope", Text: "x", Hash: "h", Embedding: []float32{1}}); err != nil {
		t.Errorf("Update missing id should be a no-op: %v", err)
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

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}
