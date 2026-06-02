package mneme

import (
	"context"
	"fmt"

	"github.com/AccursedGalaxy/mneme/provider"
	"github.com/AccursedGalaxy/mneme/store"
)

// EmbedderMismatchError reports that the configured embedder does not match the
// one recorded in the store. Searching or inserting across this boundary mixes
// vectors from two different models, which silently degrades recall — so mneme
// surfaces it as an error instead. Recover by re-embedding into a fresh store,
// or pass mneme.AllowEmbedderMismatch() to proceed anyway.
type EmbedderMismatchError struct {
	Recorded store.EmbedderInfo // what produced the stored vectors
	Current  store.EmbedderInfo // the embedder now configured
	Reason   string             // which field differs
}

func (e *EmbedderMismatchError) Error() string {
	return fmt.Sprintf("mneme: embedder mismatch (%s): store was written by %s but the "+
		"configured embedder is %s; search would silently degrade. Re-embed into a fresh "+
		"store, or pass mneme.AllowEmbedderMismatch() to override.",
		e.Reason, describeEmbedder(e.Recorded), describeEmbedder(e.Current))
}

func describeEmbedder(info store.EmbedderInfo) string {
	model := info.Model
	if model == "" {
		model = "an unnamed embedder"
	} else {
		model = fmt.Sprintf("model %q", model)
	}
	return fmt.Sprintf("%s (dim %d)", model, info.Dim)
}

// vecLen returns the dimension of the first vector in a batch, or 0 if the batch
// is empty. The store rejects zero-length embeddings, so a non-empty batch has a
// positive length.
func vecLen(vecs [][]float32) int {
	if len(vecs) == 0 {
		return 0
	}
	return len(vecs[0])
}

// embedderInfo reads the configured embedder's identity. The dimension comes
// from Dim() (which may report 0 before the first Embed call for some clients),
// and the model name from the optional provider.Named interface ("" when the
// embedder is unnamed).
func embedderInfo(e provider.Embedder) store.EmbedderInfo {
	info := store.EmbedderInfo{Dim: e.Dim()}
	if n, ok := e.(provider.Named); ok {
		info.Model = n.Name()
	}
	return info
}

// embedderMismatch reports why the current embedder is incompatible with the
// recorded identity, or "" if they are compatible. Each field is only compared
// when both sides know it: a model name is checked only when both are non-empty,
// and a dimension only when the current embedder reports a positive Dim — so an
// embedder that has not yet learned its dimension at New time is not faulted
// (the authoritative dimension check happens at insert, where it is known).
func embedderMismatch(recorded, current store.EmbedderInfo) string {
	if recorded.Model != "" && current.Model != "" && recorded.Model != current.Model {
		return "model name"
	}
	if current.Dim > 0 && recorded.Dim != current.Dim {
		return "dimension"
	}
	return ""
}

// checkEmbedderIdentity fails New when the store already records an embedder
// identity that the configured embedder does not match. A store with no recorded
// identity (fresh, or written before this guard existed) passes; the pipeline
// pins it on the first insert via recordEmbedderIdentity.
func (m *memory) checkEmbedderIdentity(ctx context.Context) error {
	if m.allowEmbedderMismatch {
		return nil
	}
	recorded, ok, err := m.store.EmbedderMeta(ctx)
	if err != nil {
		return fmt.Errorf("read embedder identity: %w", err)
	}
	if !ok {
		return nil
	}
	current := embedderInfo(m.embedder)
	if reason := embedderMismatch(recorded, current); reason != "" {
		return &EmbedderMismatchError{Recorded: recorded, Current: current, Reason: reason}
	}
	return nil
}

// recordEmbedderIdentity pins the store's embedder identity on the first insert
// and verifies it on every later one. dim is the dimension of the vectors about
// to be written — authoritative, unlike Dim() at New time. When nothing is
// recorded yet it writes the identity; when something is, it rechecks for a
// mismatch (catching, for example, an unnamed embedder whose dimension changed,
// which the New-time check could not see). Returns an *EmbedderMismatchError on
// conflict unless AllowEmbedderMismatch was set.
func (m *memory) recordEmbedderIdentity(ctx context.Context, dim int) error {
	recorded, ok, err := m.store.EmbedderMeta(ctx)
	if err != nil {
		return fmt.Errorf("read embedder identity: %w", err)
	}
	current := embedderInfo(m.embedder)
	if dim > 0 {
		current.Dim = dim
	}
	if ok {
		if m.allowEmbedderMismatch {
			return nil
		}
		if reason := embedderMismatch(recorded, current); reason != "" {
			return &EmbedderMismatchError{Recorded: recorded, Current: current, Reason: reason}
		}
		return nil
	}
	if err := m.store.SetEmbedderMeta(ctx, current); err != nil {
		return fmt.Errorf("record embedder identity: %w", err)
	}
	return nil
}
