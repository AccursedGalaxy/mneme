package mneme

import (
	"strings"
	"testing"

	"github.com/AccursedGalaxy/mneme/types"
)

func TestBuildExtractionUserSections(t *testing.T) {
	existing := []labeledMemory{
		{ID: "0", Text: "Alice works at Shopify"},
		{ID: "1", Text: "Alice owns a beagle named Max"},
	}
	context := []types.Message{{Role: "user", Content: "earlier message"}}
	conv := []types.Message{
		{Role: "user", Content: "I got promoted yesterday"},
		{Role: "assistant", Content: "Congrats!"},
		{Role: "system", Content: "should be skipped"},
	}
	out := buildExtractionUser("2026-06-01", existing, context, conv)

	for _, want := range []string{
		"OBSERVATION DATE: 2026-06-01",
		"EXISTING MEMORIES",
		"0: Alice works at Shopify",
		"1: Alice owns a beagle named Max",
		"RECENT CONTEXT",
		"earlier message",
		"NEW MESSAGES",
		"user: I got promoted yesterday",
		"assistant: Congrats!",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("builder output missing %q\n---\n%s", want, out)
		}
	}
	if strings.Contains(out, "should be skipped") {
		t.Error("system message should be skipped in rendering")
	}
}

func TestBuildExtractionUserNoExisting(t *testing.T) {
	out := buildExtractionUser("2026-06-01", nil, nil,
		[]types.Message{{Role: "user", Content: "hi"}})
	if !strings.Contains(out, "(none)") {
		t.Error("expected (none) for empty existing memories")
	}
	if strings.Contains(out, "RECENT CONTEXT") {
		t.Error("RECENT CONTEXT section should be omitted when no context given")
	}
}

func TestRenderMessagesWithName(t *testing.T) {
	out := renderMessages([]types.Message{{Role: "user", Name: "Bob", Content: "hello"}})
	if out != "user (Bob): hello" {
		t.Errorf("got %q", out)
	}
}

func TestSystemPromptVersioning(t *testing.T) {
	if systemPrompt("v1") != extractionPromptV1 {
		t.Error("v1 should map to extractionPromptV1")
	}
	if systemPrompt("does-not-exist") != extractionPromptV1 {
		t.Error("unknown version should fall back to default")
	}
	if len(PromptVersions()) == 0 {
		t.Error("expected at least one registered prompt version")
	}
}

func TestExtractionPromptCoversPrinciples(t *testing.T) {
	// Guard the load-bearing instructions so an edit can't silently drop them.
	for _, want := range []string{"Self-contained", "Specific", "One fact per item", "JSON", "attributed_to"} {
		if !strings.Contains(extractionPromptV1, want) {
			t.Errorf("extraction prompt missing principle %q", want)
		}
	}
}
