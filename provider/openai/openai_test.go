package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

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
