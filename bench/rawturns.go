package bench

import (
	"context"
	"crypto/md5"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"strings"
	"time"

	"github.com/AccursedGalaxy/mneme/provider"
	"github.com/AccursedGalaxy/mneme/store"
	"github.com/AccursedGalaxy/mneme/types"
)

// newUUID and hashRaw mirror the unexported helpers the pipeline uses to build
// records (memory.go, dedup.go). The bench harness writes records directly here,
// so it needs its own copies.
func newUUID() (string, error) {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	b[6] = (b[6] & 0x0f) | 0x40 // version 4
	b[8] = (b[8] & 0x3f) | 0x80 // variant 10
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16]), nil
}

func hashRaw(text string) string {
	sum := md5.Sum([]byte(strings.TrimSpace(text)))
	return hex.EncodeToString(sum[:])
}

// StrategyRawTurns writes the conversation itself into the store: no extraction
// LLM, no facts, just embedded turns. Search, answer and judge run unchanged, so
// a rawturns run is directly comparable to an additive one.
//
// This is the control the fact-extraction pipeline has never been measured
// against. mneme abstains on 40% of answerable LoCoMo questions because the
// answering fact was never extracted (bench/RESULTS.md, finding #2). Raw turns
// cannot lose information that way: the evidence is verbatim in the store. If
// this baseline beats additive, extraction is destroying more than it distills,
// and the fix is a hybrid store rather than another extraction prompt. If it
// loses, extraction is earning its keep and the abstentions are a recall problem
// worth attacking directly.
const StrategyRawTurns = "rawturns"

// rawTurnBatch bounds how many turn texts go into one embedding call. The
// OpenAI-compatible endpoints accept large batches, but a full LoCoMo session
// can run to hundreds of turns and an oversized request is slower to retry.
const rawTurnBatch = 128

// ingestRawTurns embeds a sample's turns and writes them straight to the store,
// bypassing the extraction LLM entirely. Turns are grouped into windows of
// `window` consecutive messages so a chunk can carry a question and its answer
// together; window=1 stores one turn per record.
func ingestRawTurns(ctx context.Context, emb provider.Embedder, st store.Store, scope types.Scope, sessions []Session, window int, now time.Time) error {
	if window <= 0 {
		window = 1
	}

	var texts []string
	for _, sess := range sessions {
		for i := 0; i < len(sess.Messages); i += window {
			end := min(i+window, len(sess.Messages))
			if chunk := renderTurns(sess, sess.Messages[i:end]); chunk != "" {
				texts = append(texts, chunk)
			}
		}
	}
	if len(texts) == 0 {
		return nil
	}

	for start := 0; start < len(texts); start += rawTurnBatch {
		end := min(start+rawTurnBatch, len(texts))
		batch := texts[start:end]

		vecs, err := emb.Embed(ctx, batch)
		if err != nil {
			return fmt.Errorf("embed raw turns: %w", err)
		}
		if len(vecs) != len(batch) {
			return fmt.Errorf("embedder returned %d vectors for %d turns", len(vecs), len(batch))
		}

		recs := make([]types.Record, len(batch))
		for i, text := range batch {
			id, err := newUUID()
			if err != nil {
				return fmt.Errorf("generate id: %w", err)
			}
			recs[i] = types.Record{
				ID:        id,
				Text:      text,
				Hash:      hashRaw(text),
				Embedding: vecs[i],
				Scope:     scope,
				CreatedAt: now,
			}
		}
		if err := st.Insert(ctx, recs); err != nil {
			return fmt.Errorf("persist raw turns: %w", err)
		}
		if start == 0 {
			// Record the embedder identity the way the pipeline would, so a
			// rawturns store is not mistaken for one written by a different
			// embedder on a later run.
			if err := st.SetEmbedderMeta(ctx, store.EmbedderInfo{Dim: len(vecs[0])}); err != nil {
				return fmt.Errorf("record embedder identity: %w", err)
			}
		}
	}
	return nil
}

// renderTurns formats a window of messages as one retrievable chunk, carrying
// the session date and speaker names inline. Both matter: LoCoMo asks "when did
// X happen" and "who said Y", and a bare turn body answers neither.
func renderTurns(sess Session, msgs []types.Message) string {
	var b strings.Builder
	if sess.Date != "" {
		fmt.Fprintf(&b, "(%s) ", sess.Date)
	}
	lines := make([]string, 0, len(msgs))
	for _, m := range msgs {
		content := strings.TrimSpace(m.Content)
		if content == "" {
			continue
		}
		if m.Name != "" {
			lines = append(lines, m.Name+": "+content)
		} else {
			lines = append(lines, content)
		}
	}
	if len(lines) == 0 {
		return ""
	}
	b.WriteString(strings.Join(lines, "\n"))
	return b.String()
}
