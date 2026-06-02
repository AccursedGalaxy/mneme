// Package provider defines the LLM and embedding interfaces mneme depends on,
// plus implementations (an OpenAI-compatible HTTP client in provider/openai and
// deterministic fakes in provider/fake).
package provider

import "context"

// LLM is the text-completion surface the extraction pipeline needs.
type LLM interface {
	// Complete returns the assistant text for the given system+user prompt.
	// When jsonObject is true the provider requests a JSON object response
	// (response_format) so the model is steered toward valid JSON; callers
	// must still parse defensively.
	Complete(ctx context.Context, system, user string, jsonObject bool) (string, error)
}

// Embedder turns text into vectors. Embed is batch: it accepts many texts and
// returns one vector per input, in order.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	Dim() int
}

// Named is an optional Embedder capability: a stable identifier for the
// embedding model (e.g. "text-embedding-3-small"). mneme uses it to pin an
// embedder identity to a store and catch accidental embedder swaps, which
// silently degrade search. Embedders that do not implement Named are identified
// by dimension alone, so swapping one unnamed embedder for a different unnamed
// one of the same dimension cannot be detected.
type Named interface {
	Name() string
}
