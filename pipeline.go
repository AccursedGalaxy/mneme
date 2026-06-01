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
	var idMap map[string]string
	if m.extractionTopK > 0 {
		qvec, err := m.embedOne(ctx, convText)
		if err != nil {
			return nil, fmt.Errorf("embed conversation: %w", err)
		}
		hits, err := m.store.Search(ctx, scope, qvec, m.extractionTopK)
		if err != nil {
			return nil, fmt.Errorf("retrieve existing memories: %w", err)
		}
		// 3. Anti-hallucination relabel: real UUIDs -> "0","1",... The reverse
		// map (integer id -> UUID) is what the consolidation pass uses to apply
		// UPDATE/DELETE back to the right record; additive mode ignores it.
		existing, idMap = relabelExisting(hits)
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

	// 6. Reconcile and persist. Consolidation needs existing facts to reconcile
	// against; with none in scope there is nothing to UPDATE/DELETE, so we save
	// the second LLM call and fall through to the additive insert.
	if m.strategy == Consolidate && len(existing) > 0 {
		return m.consolidate(ctx, scope, existing, idMap, extracted)
	}
	return m.insertNew(ctx, scope, extracted)
}

// insertNew is the additive write path (PLAN.md §4 steps 6–9): hash-dedup the
// candidates against the batch and what is already stored in scope, embed the
// survivors in one batch, and insert them. It returns the newly written facts.
// Consolidation falls back to this when its LLM response is unusable.
func (m *memory) insertNew(ctx context.Context, scope Scope, candidates []extractedFact) ([]Fact, error) {
	existingHashes, err := m.store.ExistingHashes(ctx, scope)
	if err != nil {
		return nil, fmt.Errorf("load existing hashes: %w", err)
	}
	kept, hashes := dedup(candidates, existingHashes)
	if len(kept) == 0 {
		return nil, nil
	}

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
	return facts, nil
}

// consolidate is the second LLM call of the Consolidate strategy: it asks the
// model how the candidate facts change the existing memories (ADD/UPDATE/DELETE/
// NONE), then applies those operations, mapping the prompt's integer ids back to
// real UUIDs via idMap. A malformed/empty response falls back to an additive
// insert of the candidates — never a corrupted store. It returns the facts that
// were added or updated (the changes); deletes and no-ops are not returned.
func (m *memory) consolidate(ctx context.Context, scope Scope, existing []labeledMemory, idMap map[string]string, candidates []extractedFact) ([]Fact, error) {
	system := consolidationSystemPrompt(m.consolidationVersion)
	user := buildConsolidationUser(existing, candidates)
	raw, err := m.llm.Complete(ctx, system, user, true)
	if err != nil {
		// The consolidation call failed (transient API error, empty/truncated
		// body, etc.). The candidates are already extracted, so degrade to an
		// additive insert rather than failing the caller's Add — a stale fact
		// left un-reconciled is recoverable on a later Add; a failed Add loses
		// the data. Never corrupt the store. (Same doctrine as a malformed
		// response below.)
		return m.insertNew(ctx, scope, candidates)
	}

	ops := parseConsolidation(raw)
	if len(ops) == 0 {
		// Unusable response: do no harm, behave like additive.
		return m.insertNew(ctx, scope, candidates)
	}

	// Partition ops into writes (ADD/UPDATE — both need an embedding) and
	// deletes. An UPDATE whose id does not resolve to a real record is treated
	// as an ADD: the text is still new information, and we never invent a target.
	type write struct {
		uuid string // "" => ADD (new uuid assigned at insert); else UPDATE target
		text string
	}
	var writes []write
	var deleteIDs []string
	for _, op := range ops {
		switch op.Event {
		case "ADD":
			writes = append(writes, write{text: op.Text})
		case "UPDATE":
			writes = append(writes, write{uuid: idMap[op.ID], text: op.Text}) // idMap miss => "" => ADD
		case "DELETE":
			if uuid, ok := idMap[op.ID]; ok {
				deleteIDs = append(deleteIDs, uuid)
			}
		case "NONE":
			// leave the existing memory as is
		}
	}

	// Embed all write texts in one batch.
	var facts []Fact
	if len(writes) > 0 {
		texts := make([]string, len(writes))
		for i, w := range writes {
			texts[i] = w.text
		}
		vecs, err := m.embedder.Embed(ctx, texts)
		if err != nil {
			return nil, fmt.Errorf("embed consolidated facts: %w", err)
		}
		if len(vecs) != len(writes) {
			return nil, fmt.Errorf("embedder returned %d vectors for %d writes", len(vecs), len(writes))
		}

		now := m.clock()
		var inserts []types.Record
		for i, w := range writes {
			rec := types.Record{
				Text:      w.text,
				Hash:      hashText(w.text),
				Embedding: vecs[i],
				Scope:     scope,
				CreatedAt: now,
			}
			if w.uuid == "" {
				id, err := newUUID()
				if err != nil {
					return nil, fmt.Errorf("generate id: %w", err)
				}
				rec.ID = id
				inserts = append(inserts, rec)
			} else {
				rec.ID = w.uuid
				if err := m.store.Update(ctx, rec); err != nil {
					return nil, fmt.Errorf("apply UPDATE: %w", err)
				}
			}
			facts = append(facts, recordToFact(rec, 0))
		}
		if len(inserts) > 0 {
			if err := m.store.Insert(ctx, inserts); err != nil {
				return nil, fmt.Errorf("apply ADDs: %w", err)
			}
		}
	}

	for _, id := range deleteIDs {
		if err := m.store.Delete(ctx, id); err != nil {
			return nil, fmt.Errorf("apply DELETE %s: %w", id, err)
		}
	}

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
