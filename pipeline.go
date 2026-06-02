package mneme

import (
	"context"
	"fmt"
	"sort"
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
	// incoming conversation, to show the extractor for dedup/linking. This is the
	// extractor's context only; the consolidation pass retrieves its own window
	// keyed on the extracted candidates (see retrieveForConsolidation below),
	// because the facts a new fact overturns are not the facts most like the turn.
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
		// 3. Anti-hallucination relabel: real UUIDs -> "0","1",... so the
		// extractor can reference existing memories without seeing raw ids.
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

	// 6. Reconcile and persist. Consolidation reconciles the candidates against
	// the facts they are most likely to overturn — retrieved per candidate, not
	// per conversation. With nothing reconcilable in scope there is nothing to
	// UPDATE/DELETE, so we save the second LLM call and fall through to the
	// additive insert.
	if m.strategy == Consolidate {
		conExisting, conIDMap, err := m.retrieveForConsolidation(ctx, scope, extracted)
		if err != nil {
			return nil, err
		}
		if len(conExisting) > 0 {
			return m.consolidate(ctx, scope, conExisting, conIDMap, extracted)
		}
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
	if err := m.recordEmbedderIdentity(ctx, vecLen(vecs)); err != nil {
		return nil, err
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
		hash string
	}
	var writes []write
	var deleteIDs []string

	// ADDs are hash-deduped against what is already stored in scope and against
	// each other — the same safety net insertNew applies. The prompt asks the
	// model to emit NONE for an already-known fact, but a model that ADDs a
	// duplicate anyway must not create a redundant row. UPDATEs target a specific
	// existing record, so they are never hash-deduped: overwriting in place is the
	// whole point, even when the new text happens to match another fact.
	existingHashes, err := m.store.ExistingHashes(ctx, scope)
	if err != nil {
		return nil, fmt.Errorf("load existing hashes: %w", err)
	}
	seen := make(map[string]struct{})
	addDeduped := func(text string) {
		h := hashText(text)
		if _, ok := existingHashes[h]; ok {
			return
		}
		if _, ok := seen[h]; ok {
			return
		}
		seen[h] = struct{}{}
		writes = append(writes, write{text: text, hash: h})
	}
	for _, op := range ops {
		switch op.Event {
		case "ADD":
			addDeduped(op.Text)
		case "UPDATE":
			if uuid, ok := idMap[op.ID]; ok {
				writes = append(writes, write{uuid: uuid, text: op.Text, hash: hashText(op.Text)})
			} else {
				addDeduped(op.Text) // idMap miss => new information, treat as ADD
			}
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
		if err := m.recordEmbedderIdentity(ctx, vecLen(vecs)); err != nil {
			return nil, err
		}

		now := m.clock()
		var inserts []types.Record
		for i, w := range writes {
			rec := types.Record{
				Text:      w.text,
				Hash:      w.hash,
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
				updated, err := m.store.Update(ctx, rec)
				if err != nil {
					return nil, fmt.Errorf("apply UPDATE: %w", err)
				}
				if !updated {
					// The target was concurrently deleted between retrieval and
					// this write, so Update wrote nothing. Don't claim a write
					// that didn't land — drop it from the returned facts.
					//
					// Note: the new fact text is dropped here, not re-added as an
					// ADD. Any genuinely new info in this UPDATE is lost unless the
					// same fact recurs in a later Add. We accept this because the
					// race is rare and never corrupts the store; falling back to an
					// ADD would be strictly safer but isn't worth the complexity.
					continue
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

// retrieveForConsolidation gathers the existing facts the extracted candidates
// are most likely to overturn, and relabels them for the consolidation prompt.
//
// Unlike the extractor's conversation-keyed context, this retrieves per candidate
// text — the ranking consolidation actually needs, since a candidate ("moved to
// Berlin") must surface the fact it contradicts ("lives in Munich") even when
// that fact is unlike the rest of the turn. Each candidate's neighbourhood
// (top consolidationTopK by cosine) is unioned, keeping each record's best score,
// and the union is capped at consolidationTopK so the consolidation prompt stays
// bounded regardless of candidate count. A candidate whose contradiction target
// scores below that cap can be missed — widen WithConsolidationTopK to trade
// prompt size for recall. Returns the relabelled facts and the integer-id ->
// real-UUID map the consolidation pass uses to apply UPDATE/DELETE.
func (m *memory) retrieveForConsolidation(ctx context.Context, scope Scope, candidates []extractedFact) ([]labeledMemory, map[string]string, error) {
	if m.consolidationTopK <= 0 || len(candidates) == 0 {
		return nil, nil, nil
	}

	texts := make([]string, len(candidates))
	for i, c := range candidates {
		texts[i] = c.Text
	}
	vecs, err := m.embedder.Embed(ctx, texts)
	if err != nil {
		return nil, nil, fmt.Errorf("embed candidates for consolidation: %w", err)
	}
	if len(vecs) != len(candidates) {
		return nil, nil, fmt.Errorf("embedder returned %d vectors for %d candidates", len(vecs), len(candidates))
	}

	// Union the per-candidate neighbourhoods, keeping each record's best score.
	best := make(map[string]types.Hit)
	for _, v := range vecs {
		hits, err := m.store.Search(ctx, scope, v, m.consolidationTopK)
		if err != nil {
			return nil, nil, fmt.Errorf("retrieve for consolidation: %w", err)
		}
		for _, h := range hits {
			if prev, ok := best[h.ID]; !ok || h.Score > prev.Score {
				best[h.ID] = h
			}
		}
	}
	if len(best) == 0 {
		return nil, nil, nil
	}

	// Rank by score (ties broken by id for a deterministic relabel) and cap.
	merged := make([]types.Hit, 0, len(best))
	for _, h := range best {
		merged = append(merged, h)
	}
	sort.Slice(merged, func(i, j int) bool {
		if merged[i].Score != merged[j].Score {
			return merged[i].Score > merged[j].Score
		}
		return merged[i].ID < merged[j].ID
	})
	if len(merged) > m.consolidationTopK {
		merged = merged[:m.consolidationTopK]
	}

	labeled, idMap := relabelExisting(merged)
	return labeled, idMap, nil
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
