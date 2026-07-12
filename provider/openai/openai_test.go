package openai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// fastRetries shrinks the backoff schedule so retry-loop tests run in
// milliseconds instead of sleeping through the production delays.
func fastRetries(t *testing.T) {
	t.Helper()
	prev := retryBaseDelay
	retryBaseDelay = time.Millisecond
	t.Cleanup(func() { retryBaseDelay = prev })
}

func TestLLMComplete(t *testing.T) {
	var gotBody chatRequest
	var gotAuth, gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotPath = r.URL.Path
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &gotBody)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"hello back"}}]}`)
	}))
	defer srv.Close()

	l := &LLM{BaseURL: srv.URL, APIKey: "sk-test", Model: "gpt-4o-mini"}
	got, err := l.Complete(context.Background(), "you are terse", "hi", true)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if got != "hello back" {
		t.Errorf("content = %q", got)
	}
	if gotAuth != "Bearer sk-test" {
		t.Errorf("auth header = %q", gotAuth)
	}
	if gotPath != "/chat/completions" {
		t.Errorf("path = %q", gotPath)
	}
	if gotBody.Model != "gpt-4o-mini" {
		t.Errorf("model = %q", gotBody.Model)
	}
	if gotBody.ResponseFormat == nil || gotBody.ResponseFormat.Type != "json_object" {
		t.Errorf("expected json_object response_format, got %+v", gotBody.ResponseFormat)
	}
	if len(gotBody.Messages) != 2 || gotBody.Messages[0].Role != "system" || gotBody.Messages[1].Content != "hi" {
		t.Errorf("messages malformed: %+v", gotBody.Messages)
	}
}

func TestLLMNoSystemMessage(t *testing.T) {
	var gotBody chatRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &gotBody)
		io.WriteString(w, `{"choices":[{"message":{"content":"ok"}}]}`)
	}))
	defer srv.Close()
	l := &LLM{BaseURL: srv.URL, Model: "m"}
	if _, err := l.Complete(context.Background(), "", "just user", false); err != nil {
		t.Fatal(err)
	}
	if len(gotBody.Messages) != 1 || gotBody.Messages[0].Role != "user" {
		t.Errorf("expected single user message, got %+v", gotBody.Messages)
	}
	if gotBody.ResponseFormat != nil {
		t.Errorf("expected no response_format when jsonObject=false")
	}
}

func TestLLMErrorStatus(t *testing.T) {
	fastRetries(t)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		io.WriteString(w, `{"error":{"message":"rate limited"}}`)
	}))
	defer srv.Close()
	l := &LLM{BaseURL: srv.URL, Model: "m"}
	_, err := l.Complete(context.Background(), "", "x", false)
	if err == nil || !strings.Contains(err.Error(), "429") {
		t.Errorf("expected 429 error, got %v", err)
	}
}

func TestRetryExhausts429(t *testing.T) {
	fastRetries(t)
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusTooManyRequests)
	}))
	defer srv.Close()
	l := &LLM{BaseURL: srv.URL, Model: "m"}
	_, err := l.Complete(context.Background(), "", "x", false)
	if err == nil || !strings.Contains(err.Error(), "giving up after 5 attempts") {
		t.Errorf("expected giving-up error, got %v", err)
	}
	if got := calls.Load(); got != 5 {
		t.Errorf("429 should be retried maxAttempts times, got %d requests", got)
	}
}

func TestRetryRecoversAfter5xx(t *testing.T) {
	fastRetries(t)
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) < 3 {
			w.WriteHeader(http.StatusBadGateway)
			return
		}
		io.WriteString(w, `{"choices":[{"message":{"content":"ok"}}]}`)
	}))
	defer srv.Close()
	l := &LLM{BaseURL: srv.URL, Model: "m"}
	got, err := l.Complete(context.Background(), "", "x", false)
	if err != nil {
		t.Fatalf("should recover after transient 5xx: %v", err)
	}
	if got != "ok" || calls.Load() != 3 {
		t.Errorf("got %q after %d calls, want ok after 3", got, calls.Load())
	}
}

func TestNoRetryOn400(t *testing.T) {
	fastRetries(t)
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, `{"error":{"message":"bad prompt"}}`)
	}))
	defer srv.Close()
	l := &LLM{BaseURL: srv.URL, Model: "m"}
	if _, err := l.Complete(context.Background(), "", "x", false); err == nil {
		t.Fatal("expected error on 400")
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("a non-429 4xx must not be retried, got %d requests", got)
	}
}

func TestRetryGarbled2xxBody(t *testing.T) {
	fastRetries(t)
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			io.WriteString(w, `not json at all`) // gateway hiccup: 200 + garbage
			return
		}
		io.WriteString(w, `{"choices":[{"message":{"content":"ok"}}]}`)
	}))
	defer srv.Close()
	l := &LLM{BaseURL: srv.URL, Model: "m"}
	got, err := l.Complete(context.Background(), "", "x", false)
	if err != nil || got != "ok" {
		t.Errorf("garbled 2xx body should be retried: got %q, %v", got, err)
	}
}

func TestRetryContextCancelKeepsCause(t *testing.T) {
	fastRetries(t)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		io.WriteString(w, "overloaded")
	}))
	defer srv.Close()
	// Restore the real backoff for this test only, so cancellation reliably
	// lands during the wait rather than racing the next request.
	retryBaseDelay = 250 * time.Millisecond
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(20 * time.Millisecond) // after the first 503, during backoff
		cancel()
	}()
	// Cancellation during a backoff wait must produce an error carrying both
	// the cancellation and the failure that caused the retrying.
	l := &LLM{BaseURL: srv.URL, Model: "m"}
	_, err := l.Complete(ctx, "", "x", false)
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("error should wrap context.Canceled: %v", err)
	}
	if !strings.Contains(err.Error(), "503") {
		t.Errorf("error should preserve the retry cause (503): %v", err)
	}
}

func TestTemperatureRejectionRetriesWithout(t *testing.T) {
	fastRetries(t)
	var bodies []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		bodies = append(bodies, string(body))
		if strings.Contains(string(body), "temperature") {
			w.WriteHeader(http.StatusBadRequest)
			io.WriteString(w, `{"error":{"message":"'temperature' is not supported with this model"}}`)
			return
		}
		io.WriteString(w, `{"choices":[{"message":{"content":"ok"}}]}`)
	}))
	defer srv.Close()
	l := &LLM{BaseURL: srv.URL, Model: "o-mini"}
	got, err := l.Complete(context.Background(), "", "x", false)
	if err != nil {
		t.Fatalf("should retry without temperature after a rejection: %v", err)
	}
	if got != "ok" {
		t.Errorf("content = %q", got)
	}
	if len(bodies) != 2 || !strings.Contains(bodies[0], "temperature") || strings.Contains(bodies[1], "temperature") {
		t.Errorf("expected temp then temp-free request, got %d bodies: %v", len(bodies), bodies)
	}
}

func TestEmbedderBatchAndOrder(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" {
			t.Errorf("path = %q", r.URL.Path)
		}
		var req embedRequest
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &req)
		// Return out-of-order data to verify reordering by index.
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"data":[
			{"index":1,"embedding":[0.4,0.5,0.6]},
			{"index":0,"embedding":[0.1,0.2,0.3]}
		]}`)
	}))
	defer srv.Close()

	e := &Embedder{BaseURL: srv.URL, Model: "text-embedding-3-small"}
	vecs, err := e.Embed(context.Background(), []string{"first", "second"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vecs) != 2 {
		t.Fatalf("want 2 vectors, got %d", len(vecs))
	}
	if vecs[0][0] != 0.1 || vecs[1][0] != 0.4 {
		t.Errorf("reordering by index failed: %v", vecs)
	}
	if e.Dim() != 3 {
		t.Errorf("Dim learned = %d, want 3", e.Dim())
	}
}

func TestEmbedderFixedDimBeforeCall(t *testing.T) {
	e := &Embedder{FixedDim: 1536}
	if e.Dim() != 1536 {
		t.Errorf("Dim = %d, want 1536", e.Dim())
	}
}

func TestEmbedderEmptyInput(t *testing.T) {
	e := &Embedder{BaseURL: "http://unused", Model: "m"}
	vecs, err := e.Embed(context.Background(), nil)
	if err != nil || vecs != nil {
		t.Errorf("empty input should no-op: vecs=%v err=%v", vecs, err)
	}
}

func TestEmbedderRejectsEmptyVector(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"data":[{"index":0,"embedding":[0.1]},{"index":1,"embedding":[]}]}`)
	}))
	defer srv.Close()
	e := &Embedder{BaseURL: srv.URL, Model: "m"}
	_, err := e.Embed(context.Background(), []string{"a", "b"})
	if err == nil || !strings.Contains(err.Error(), "empty vector") {
		t.Errorf("an empty embedding must be rejected, got %v", err)
	}
}

func TestEmbedderCountMismatch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"data":[{"index":0,"embedding":[0.1]}]}`)
	}))
	defer srv.Close()
	e := &Embedder{BaseURL: srv.URL, Model: "m"}
	_, err := e.Embed(context.Background(), []string{"a", "b"})
	if err == nil || !strings.Contains(err.Error(), "1 vectors for 2") {
		t.Errorf("expected count-mismatch error, got %v", err)
	}
}
