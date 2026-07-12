// Package openai is a minimal OpenAI-compatible HTTP client implementing
// provider.LLM and provider.Embedder over net/http with no SDK dependency. It
// targets the /chat/completions and /embeddings endpoints, so it works against
// OpenAI, Ollama, vLLM, LM Studio and anything else speaking that wire format.
package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/AccursedGalaxy/mneme/provider"
)

// DefaultTimeout is used when a client is created without its own *http.Client.
const DefaultTimeout = 60 * time.Second

func defaultHTTP() *http.Client { return &http.Client{Timeout: DefaultTimeout} }

func trimBase(u string) string { return strings.TrimRight(u, "/") }

// maxAttempts bounds how many times doJSON tries a single request. Transient
// failures — network errors, 429/5xx, and the empty/truncated bodies a busy
// gateway (e.g. OpenRouter under load) returns as a 2xx with no JSON — are
// retried with exponential backoff. Without this a single blip aborts a
// multi-thousand-call benchmark or ingestion run, which is not acceptable for a
// memory layer meant to run against real, occasionally-flaky endpoints.
const maxAttempts = 5

// retryBaseDelay is the first backoff step (doubled each attempt: 250ms, 500ms,
// 1s, 2s by default). A variable, not a constant, so tests can exercise the
// retry loop without sleeping through the real schedule.
var retryBaseDelay = 250 * time.Millisecond

// doJSON posts body as JSON to baseURL+path and decodes the response into out.
// It returns a descriptive error on non-2xx, including the response body, and
// retries transient failures with exponential backoff (see maxAttempts).
func doJSON(ctx context.Context, hc *http.Client, baseURL, apiKey, path string, body, out any) error {
	buf, err := json.Marshal(body)
	if err != nil {
		return err
	}
	url := trimBase(baseURL) + path

	var lastErr error
	for attempt := 0; attempt < maxAttempts; attempt++ {
		if attempt > 0 {
			// Exponential backoff: 250ms, 500ms, 1s, 2s. Abort early if the
			// caller's context is cancelled while we wait — but keep the error
			// that caused the retrying, or the caller sees a bare deadline
			// error with zero diagnostic about why time was being spent.
			delay := retryBaseDelay << (attempt - 1)
			select {
			case <-ctx.Done():
				return fmt.Errorf("%w (while retrying after: %v)", ctx.Err(), lastErr)
			case <-time.After(delay):
			}
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(buf))
		if err != nil {
			return err // request construction can't be fixed by retrying
		}
		req.Header.Set("Content-Type", "application/json")
		if apiKey != "" {
			req.Header.Set("Authorization", "Bearer "+apiKey)
		}

		resp, err := hc.Do(req)
		if err != nil {
			// Network/transport error (reset, timeout, EOF). Don't retry a
			// cancelled context — that's the caller's intent, not a blip.
			if ctx.Err() != nil {
				return err
			}
			lastErr = fmt.Errorf("openai %s: %w", path, err)
			continue
		}
		data, readErr := io.ReadAll(resp.Body)
		resp.Body.Close()
		if readErr != nil {
			// A body cut off mid-read is a transport blip; a truncated prefix
			// that happens to be valid JSON must not be accepted as complete.
			lastErr = fmt.Errorf("openai %s: read response: %w", path, readErr)
			continue
		}

		if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500 {
			lastErr = fmt.Errorf("openai %s: status %d: %s", path, resp.StatusCode, strings.TrimSpace(string(data)))
			continue // 429 / 5xx are transient — back off and retry
		}
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			// 4xx other than 429 is a client error; retrying won't help.
			return fmt.Errorf("openai %s: status %d: %s", path, resp.StatusCode, strings.TrimSpace(string(data)))
		}
		if err := json.Unmarshal(data, out); err != nil {
			// A 2xx with an empty/garbled body is a gateway hiccup, not a real
			// response — retry rather than fail the whole run.
			lastErr = fmt.Errorf("openai %s: decode response: %w", path, err)
			continue
		}
		return nil
	}
	return fmt.Errorf("openai %s: giving up after %d attempts: %w", path, maxAttempts, lastErr)
}

// LLM is an OpenAI-compatible chat-completions client.
type LLM struct {
	BaseURL string
	APIKey  string
	Model   string
	HTTP    *http.Client
}

var _ provider.LLM = (*LLM)(nil)

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model          string          `json:"model"`
	Messages       []chatMessage   `json:"messages"`
	ResponseFormat *responseFormat `json:"response_format,omitempty"`
	// Temperature is a pointer so it can be dropped from the request entirely:
	// reasoning-family models reject an explicit temperature with a 400 (see
	// the fallback in Complete).
	Temperature *float32 `json:"temperature,omitempty"`
}

type responseFormat struct {
	Type string `json:"type"`
}

type chatResponse struct {
	Choices []struct {
		Message chatMessage `json:"message"`
	} `json:"choices"`
	Error *apiError `json:"error,omitempty"`
}

type apiError struct {
	Message string `json:"message"`
}

func (l *LLM) Complete(ctx context.Context, system, user string, jsonObject bool) (string, error) {
	hc := l.HTTP
	if hc == nil {
		hc = defaultHTTP()
	}
	var msgs []chatMessage
	if system != "" {
		msgs = append(msgs, chatMessage{Role: "system", Content: system})
	}
	msgs = append(msgs, chatMessage{Role: "user", Content: user})

	temp := float32(0) // deterministic extraction by default
	reqBody := chatRequest{Model: l.Model, Messages: msgs, Temperature: &temp}
	if jsonObject {
		reqBody.ResponseFormat = &responseFormat{Type: "json_object"}
	}

	var out chatResponse
	err := doJSON(ctx, hc, l.BaseURL, l.APIKey, "/chat/completions", reqBody, &out)
	if err != nil && isTemperatureRejection(err) {
		// Reasoning-family models (OpenAI o-series and successors) reject an
		// explicit temperature parameter with a 400. Their decoding is not
		// tunable anyway, so dropping the parameter loses nothing — retry once
		// without it instead of hard-failing every call against such a model.
		reqBody.Temperature = nil
		out = chatResponse{}
		err = doJSON(ctx, hc, l.BaseURL, l.APIKey, "/chat/completions", reqBody, &out)
	}
	if err != nil {
		return "", err
	}
	if out.Error != nil {
		return "", fmt.Errorf("openai chat: %s", out.Error.Message)
	}
	if len(out.Choices) == 0 {
		return "", fmt.Errorf("openai chat: no choices returned")
	}
	return out.Choices[0].Message.Content, nil
}

// Embedder is an OpenAI-compatible embeddings client. The vector dimension is
// learned from the first successful Embed call and cached for Dim().
type Embedder struct {
	BaseURL string
	APIKey  string
	Model   string
	HTTP    *http.Client
	// FixedDim, if > 0, is reported by Dim() without needing a call first.
	FixedDim int

	mu  sync.Mutex
	dim int
}

var _ provider.Embedder = (*Embedder)(nil)

type embedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type embedResponse struct {
	Data []struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
	Error *apiError `json:"error,omitempty"`
}

func (e *Embedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	hc := e.HTTP
	if hc == nil {
		hc = defaultHTTP()
	}
	var out embedResponse
	if err := doJSON(ctx, hc, e.BaseURL, e.APIKey, "/embeddings",
		embedRequest{Model: e.Model, Input: texts}, &out); err != nil {
		return nil, err
	}
	if out.Error != nil {
		return nil, fmt.Errorf("openai embeddings: %s", out.Error.Message)
	}
	if len(out.Data) != len(texts) {
		return nil, fmt.Errorf("openai embeddings: got %d vectors for %d inputs", len(out.Data), len(texts))
	}
	// Order by index — the API may not preserve request order.
	vecs := make([][]float32, len(texts))
	for _, d := range out.Data {
		if d.Index < 0 || d.Index >= len(vecs) {
			return nil, fmt.Errorf("openai embeddings: out-of-range index %d", d.Index)
		}
		vecs[d.Index] = d.Embedding
	}
	for i, v := range vecs {
		// A missing entry or an "embedding":[] both mean the response is
		// unusable: an empty vector persisted downstream would score 0 against
		// every query forever, so reject the batch loudly instead.
		if len(v) == 0 {
			return nil, fmt.Errorf("openai embeddings: missing or empty vector at index %d", i)
		}
	}
	if len(vecs[0]) > 0 {
		e.mu.Lock()
		e.dim = len(vecs[0])
		e.mu.Unlock()
	}
	return vecs, nil
}

func (e *Embedder) Dim() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.dim > 0 {
		return e.dim
	}
	return e.FixedDim
}

// Name reports the embedding model, satisfying provider.Named so mneme can pin
// this embedder's identity to a store and detect an accidental swap.
func (e *Embedder) Name() string { return e.Model }

// isTemperatureRejection reports whether err looks like a 400 caused by the
// model refusing an explicit temperature parameter (the wire error message
// names the offending parameter). Heuristic on purpose: the OpenAI-compatible
// ecosystem has no structured error code for this.
func isTemperatureRejection(err error) bool {
	s := strings.ToLower(err.Error())
	return strings.Contains(s, "status 400") && strings.Contains(s, "temperature")
}
