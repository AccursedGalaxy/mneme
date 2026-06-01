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
