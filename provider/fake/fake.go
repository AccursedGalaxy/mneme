// Package fake provides deterministic, offline implementations of provider.LLM
// and provider.Embedder for unit tests. No network, no randomness: the same
// input always yields the same output, so tests are reproducible.
package fake

import (
	"context"
	"fmt"
	"hash/fnv"
	"math"
	"strings"
	"sync"

	"github.com/AccursedGalaxy/mneme/provider"
)

// Embedder is a deterministic feature-hashing embedder. It lowercases input,
// splits on non-alphanumeric runs, and hashes each token into one of Dim
// buckets (with a sign), then L2-normalizes. Texts that share words end up with
// similar vectors, so it is good enough to exercise the real search path
// offline — unlike pure random vectors, nearest-neighbour order is meaningful.
type Embedder struct {
	D int // dimension; defaults to 64 when zero
}

var _ provider.Embedder = (*Embedder)(nil)

func (e *Embedder) dim() int {
	if e.D <= 0 {
		return 64
	}
	return e.D
}

func (e *Embedder) Dim() int { return e.dim() }

func (e *Embedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = e.embedOne(t)
	}
	return out, nil
}

func (e *Embedder) embedOne(text string) []float32 {
	d := e.dim()
	v := make([]float32, d)
	for _, tok := range tokenize(text) {
		h := fnv.New32a()
		h.Write([]byte(tok))
		sum := h.Sum32()
		bucket := int(sum % uint32(d))
		sign := float32(1)
		if sum&1 == 1 {
			sign = -1
		}
		v[bucket] += sign
	}
	// L2 normalize so cosine is stable and self-similarity is 1.
	var norm float64
	for _, x := range v {
		norm += float64(x) * float64(x)
	}
	if norm > 0 {
		inv := float32(1 / math.Sqrt(norm))
		for i := range v {
			v[i] *= inv
		}
	}
	return v
}

func tokenize(s string) []string {
	raw := strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !((r >= 'a' && r <= 'z') || (r >= '0' && r <= '9'))
	})
	out := make([]string, len(raw))
	for i, t := range raw {
		out[i] = stem(t)
	}
	return out
}

// stem applies a crude suffix strip so lexical variants collide (works->work,
// adopted->adopt, beagles->beagle). It is deliberately naive — just enough to
// make the lexical fake useful for offline search tests, not a real stemmer.
func stem(t string) string {
	for _, suf := range []string{"ing", "ed", "es", "s"} {
		if len(t) > len(suf)+2 && strings.HasSuffix(t, suf) {
			return t[:len(t)-len(suf)]
		}
	}
	return t
}

// Call records one invocation of LLM.Complete, for test assertions.
type Call struct {
	System     string
	User       string
	JSONObject bool
}

// LLM is a scriptable provider.LLM. Set Responder to compute a reply from the
// prompt, or leave it nil and set Responses to return canned replies in order
// (falling back to Default once exhausted). Calls records every invocation.
type LLM struct {
	// Responder, when non-nil, fully controls the reply.
	Responder func(system, user string) (string, error)
	// Responses are returned in order when Responder is nil.
	Responses []string
	// Default is returned when Responder is nil and Responses is exhausted.
	Default string

	mu    sync.Mutex
	Calls []Call
	next  int
}

var _ provider.LLM = (*LLM)(nil)

func (l *LLM) Complete(_ context.Context, system, user string, jsonObject bool) (string, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.Calls = append(l.Calls, Call{System: system, User: user, JSONObject: jsonObject})

	if l.Responder != nil {
		return l.Responder(system, user)
	}
	if l.next < len(l.Responses) {
		r := l.Responses[l.next]
		l.next++
		return r, nil
	}
	return l.Default, nil
}

// CallCount returns how many times Complete has been called.
func (l *LLM) CallCount() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return len(l.Calls)
}

// JSON is a tiny helper to build a canned extraction response from fact texts.
// It produces {"memory":[{"id":"0","text":"...","attributed_to":"user"},...]}.
func JSON(facts ...string) string {
	var b strings.Builder
	b.WriteString(`{"memory":[`)
	for i, f := range facts {
		if i > 0 {
			b.WriteByte(',')
		}
		fmt.Fprintf(&b, `{"id":"%d","text":%q,"attributed_to":"user"}`, i, f)
	}
	b.WriteString("]}")
	return b.String()
}
