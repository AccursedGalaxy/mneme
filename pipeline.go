package mneme

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/AccursedGalaxy/mneme/types"
)

// Add runs the additive extraction pipeline (PLAN.md §4): retrieve existing
// memories for context, extract durable facts with one LLM call, embed and
// dedup them, and persist the survivors. It returns the newly written facts.
//
// A malformed or empty LLM response is not an error: it yields zero new facts
// and a nil error, so a bad model turn never fails the caller's Add.
func (m *memory) Add(ctx context.Context, msgs []Message, scope Scope) ([]Fact, error) {
	convText := conversationText(msgs)
	if convText == "" {
		return nil, nil // nothing to extract from
	}

	// 1+2. Retrieve existing memories in scope, ranked by relevance to the
	// incoming conversation, to show the extractor for dedup/linking.
	var existing []labeledMemory
	if m.extractionTopK > 0 {
		qvec, err := m.embedOne(ctx, convText)
		if err != nil {
			return nil, fmt.Errorf("embed conversation: %w", err)
		}
		hits, err := m.store.Search(ctx, scope, qvec, m.extractionTopK)
		if err != nil {
			return nil, fmt.Errorf("retrieve existing memories: %w", err)
		}
		// 3. Anti-hallucination relabel: real UUIDs -> "0","1",... The map is
		// kept even though additive mode does not reference existing ids back,
		// preserving the seam for a future update/delete pass.
		existing, _ = relabelExisting(hits)
	}

	// 4. Extract (one LLM call) with our versioned prompt.
	system := systemPrompt(m.promptVersion)
	user := buildExtractionUser(m.today(), existing, nil, msgs)
	raw, err := m.llm.Complete(ctx, system, user, true)
	if err != nil {
		return nil, fmt.Errorf("extraction LLM call: %w", err)
	}

	// 5. Parse defensively — never panic/error on bad JSON.
	extracted := parseExtraction(raw)
	if len(extracted) == 0 {
		return nil, nil
	}

	// 7. Dedup by hash against the batch and what is already stored in scope.
	existingHashes, err := m.store.ExistingHashes(ctx, scope)
	if err != nil {
		return nil, fmt.Errorf("load existing hashes: %w", err)
	}
	kept, hashes := dedup(extracted, existingHashes)
	if len(kept) == 0 {
		return nil, nil
	}

	// 6. Embed the survivors in one batch.
	texts := make([]string, len(kept))
	for i, f := range kept {
		texts[i] = f.Text
	}
	vecs, err := m.embedder.Embed(ctx, texts)
	if err != nil {
		return nil, fmt.Errorf("embed extracted facts: %w", err)
	}
	if len(vecs) != len(kept) {
		return nil, fmt.Errorf("embedder returned %d vectors for %d facts", len(vecs), len(kept))
	}

	// 8. Persist.
	now := m.clock()
	recs := make([]types.Record, len(kept))
	facts := make([]Fact, len(kept))
	for i, f := range kept {
		id, err := newUUID()
		if err != nil {
			return nil, fmt.Errorf("generate id: %w", err)
		}
		recs[i] = types.Record{
			ID:        id,
			Text:      f.Text,
			Hash:      hashes[i],
			Embedding: vecs[i],
			Scope:     scope,
			CreatedAt: now,
		}
		facts[i] = recordToFact(recs[i], 0)
	}
	if err := m.store.Insert(ctx, recs); err != nil {
		return nil, fmt.Errorf("persist facts: %w", err)
	}

	// 9. Return the newly written facts.
	return facts, nil
}

// Search embeds the query and returns the top-k facts in scope by cosine
// similarity, Score populated, highest first.
func (m *memory) Search(ctx context.Context, query string, scope Scope, k int) ([]Fact, error) {
	if strings.TrimSpace(query) == "" || k <= 0 {
		return nil, nil
	}
	qvec, err := m.embedOne(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}
	hits, err := m.store.Search(ctx, scope, qvec, k)
	if err != nil {
		return nil, fmt.Errorf("search: %w", err)
	}
	facts := make([]Fact, len(hits))
	for i, h := range hits {
		facts[i] = recordToFact(h.Record, h.Score)
	}
	return facts, nil
}

// embedOne embeds a single string and returns its vector.
func (m *memory) embedOne(ctx context.Context, text string) ([]float32, error) {
	vecs, err := m.embedder.Embed(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(vecs) != 1 {
		return nil, fmt.Errorf("embedder returned %d vectors for 1 input", len(vecs))
	}
	return vecs[0], nil
}

// today formats the pipeline's clock for date grounding in the prompt.
func (m *memory) today() string {
	return m.clock().Format("2006-01-02 (Monday)")
}

// relabelExisting maps retrieved facts' real UUIDs to small integer-string ids
// for the prompt, returning the integer-labelled memories and the reverse map
// (integer id -> real UUID).
func relabelExisting(hits []types.Hit) ([]labeledMemory, map[string]string) {
	labels := make([]labeledMemory, len(hits))
	idMap := make(map[string]string, len(hits))
	for i, h := range hits {
		intID := strconv.Itoa(i)
		labels[i] = labeledMemory{ID: intID, Text: h.Text}
		idMap[intID] = h.ID
	}
	return labels, idMap
}

// conversationText flattens the extractable (non-system, non-empty) messages
// into one string for embedding the incoming turn.
func conversationText(msgs []Message) string {
	return renderMessages(msgs)
}
