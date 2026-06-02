package mneme

import (
	"context"
	"fmt"
	"sort"
	"strings"
)

// Reranker reorders a pool of retrieved candidates by relevance to the query,
// so the facts most likely to answer it are pulled into the top-k window that
// brute-force cosine over a single phrasing can dilute. Search retrieves a wider
// pool (see DefaultRerankPoolN), hands it here, and keeps the leading k of
// whatever order this returns.
//
// Implementations should be defensive: a Rerank that cannot improve on the
// input order (e.g. a malformed model response) ought to return the candidates
// unchanged rather than error. A returned error propagates out of Search.
//
// The interface is deliberately tiny so a future cross-encoder or hosted rerank
// API can drop in beside the shipped LLM reranker (provider/openai.LLMReranker).
type Reranker interface {
	Rerank(ctx context.Context, query string, candidates []Fact) ([]Fact, error)
}

// DefaultRerankPoolN is how many candidates Search retrieves per query before a
// reranker (or the multi-query union) narrows them to k. Over-retrieving is the
// whole point of a booster: the answer fact often exists but sits at rank 8-15
// under raw cosine, so a k=5 cutoff drops it. ~20 is wide enough to recover
// those without making the rerank prompt unwieldy.
const DefaultRerankPoolN = 20

// WithReranker enables a reranking pass inside Search: candidates are retrieved
// at DefaultRerankPoolN depth, reordered by r, then truncated to k. The public
// Search signature is unchanged. Without it, Search returns the top-k by raw
// cosine as before.
func WithReranker(r Reranker) Option { return func(m *memory) { m.reranker = r } }

// WithMultiQuery makes Search expand the query into up to n diverse search
// phrasings via one LLM call, retrieve each, and union the hits (dedup by id,
// best score wins) before reranking. It targets multi-hop questions, where a
// single phrasing won't surface every fact needed to answer. n <= 1 disables
// expansion (the original query is always used). Expansion is best-effort: if
// the LLM call or its parse fails, Search falls back to the original query
// alone rather than erroring.
func WithMultiQuery(n int) Option {
	return func(m *memory) {
		if n >= 0 {
			m.multiQueryN = n
		}
	}
}

// rerankPoolN is the retrieval depth Search uses when a booster (reranker or
// multi-query) is active: wide enough to recover diluted facts, but never below
// k when the caller asks for more than the default.
func rerankPoolN(k int) int {
	if k > DefaultRerankPoolN {
		return k
	}
	return DefaultRerankPoolN
}

// searchQueries returns the query phrasings Search will retrieve against. With
// multi-query disabled (or no LLM), that is just the original. Otherwise it asks
// the LLM for alternatives and returns a deduped list, original first, capped at
// multiQueryN. Best-effort: any failure degrades to the original query alone.
func (m *memory) searchQueries(ctx context.Context, query string) []string {
	if m.multiQueryN <= 1 || m.llm == nil {
		return []string{query}
	}
	raw, err := m.llm.Complete(ctx, multiQueryPromptV1, buildMultiQueryUser(query, m.multiQueryN), true)
	if err != nil {
		return []string{query}
	}
	return mergeQueries(query, parseQueries(raw), m.multiQueryN)
}

// mergeQueries builds the final query list: the original always first, then the
// expansions, deduped case-insensitively and capped at n.
func mergeQueries(original string, expansions []string, n int) []string {
	out := []string{original}
	seen := map[string]struct{}{strings.ToLower(strings.TrimSpace(original)): {}}
	for _, q := range expansions {
		if len(out) >= n {
			break
		}
		key := strings.ToLower(strings.TrimSpace(q))
		if key == "" {
			continue
		}
		if _, dup := seen[key]; dup {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, q)
	}
	return out
}

// gatherCandidates retrieves poolN hits for each query and unions them by fact
// id, keeping each fact's best score across phrasings, then returns them in a
// deterministic order (score desc, newest first, id asc) — the same order
// store.Search uses, so a single-query no-booster call is byte-identical to the
// pre-booster path.
func (m *memory) gatherCandidates(ctx context.Context, queries []string, scope Scope, poolN int) ([]Fact, error) {
	best := make(map[string]Fact)
	for _, q := range queries {
		qvec, err := m.embedOne(ctx, q)
		if err != nil {
			return nil, fmt.Errorf("embed query: %w", err)
		}
		hits, err := m.store.Search(ctx, scope, qvec, poolN)
		if err != nil {
			return nil, fmt.Errorf("search: %w", err)
		}
		for _, h := range hits {
			f := recordToFact(h.Record, h.Score)
			if cur, ok := best[f.ID]; !ok || f.Score > cur.Score {
				best[f.ID] = f
			}
		}
	}

	facts := make([]Fact, 0, len(best))
	for _, f := range best {
		facts = append(facts, f)
	}
	sortFactsByScore(facts)
	return facts, nil
}

// sortFactsByScore orders facts highest-score first, breaking ties by newest
// CreatedAt then ascending id — the same total order store.Search imposes, so
// retrieval stays deterministic across the union.
func sortFactsByScore(facts []Fact) {
	sort.Slice(facts, func(i, j int) bool {
		if facts[i].Score != facts[j].Score {
			return facts[i].Score > facts[j].Score
		}
		if !facts[i].CreatedAt.Equal(facts[j].CreatedAt) {
			return facts[i].CreatedAt.After(facts[j].CreatedAt)
		}
		return facts[i].ID < facts[j].ID
	})
}

// multiQueryPromptV1 is mneme's own query-expansion system prompt. Like the
// extraction and consolidation prompts it is owned IP, phrased in our words.
const multiQueryPromptV1 = `You rewrite a search query into several alternative phrasings to improve recall over a long-term memory store.

Given a QUESTION, produce diverse alternative search queries a semantic search could use to find the facts needed to answer it. Vary the wording, expand abbreviations, and when the question has several parts, split it so each needed fact is reachable by at least one query.

RULES
- Keep every proper noun, name, date and quantity from the question.
- Each query is a standalone search phrase, not a question addressed to the user.
- Do not answer the question; only rephrase it for retrieval.
- Do not repeat the original question verbatim — offer genuinely different angles.

OUTPUT
Return ONLY a JSON object, no prose and no code fences, in exactly this shape:
{"queries":["<a search phrase>","<another search phrase>"]}`

// buildMultiQueryUser is the per-call half of the expansion prompt: the question
// and how many alternatives to produce.
func buildMultiQueryUser(query string, n int) string {
	return fmt.Sprintf("QUESTION: %s\n\nProduce up to %d alternative search queries.", query, n-1)
}
