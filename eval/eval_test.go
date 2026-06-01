package eval

import (
	"context"
	"strings"
	"testing"

	"github.com/AccursedGalaxy/mneme/provider/fake"
	"github.com/AccursedGalaxy/mneme/store/sqlite"
	"github.com/AccursedGalaxy/mneme/types"
)

func msg(role, content string) []types.Message {
	return []types.Message{{Role: role, Content: content}}
}

func TestLoadFixtures(t *testing.T) {
	fixtures, err := LoadFixtures("fixtures")
	if err != nil {
		t.Fatalf("LoadFixtures: %v", err)
	}
	if len(fixtures) < 15 {
		t.Fatalf("expected >=15 fixtures (PLAN §6), got %d", len(fixtures))
	}
	// Spot-check structure and that the documented categories are present.
	names := map[string]bool{}
	var sawNothing, sawMultiTopic, sawSpecificity bool
	for _, f := range fixtures {
		if f.Name == "" {
			t.Error("fixture missing name")
		}
		names[f.Name] = true
		if len(f.ExpectedFacts) == 0 {
			sawNothing = true
		}
		if len(f.ExpectedFacts) >= 3 {
			sawMultiTopic = true
		}
		if len(f.RequiredTokens) > 0 {
			sawSpecificity = true
		}
	}
	if !sawNothing {
		t.Error("no nothing-to-extract fixture present")
	}
	if !sawMultiTopic {
		t.Error("no multi-topic (>=3 facts) fixture present")
	}
	if !sawSpecificity {
		t.Error("no specificity (required_tokens) fixture present")
	}
}

func TestSpecificityDeterministic(t *testing.T) {
	extracted := []string{"The user owns a 2022 Ferrari 488 GTB in rosso red"}
	if got := specificity([]string{"Ferrari 488 GTB", "2022", "rosso"}, extracted); got != 1 {
		t.Errorf("all tokens present should score 1, got %v", got)
	}
	if got := specificity([]string{"Lamborghini"}, extracted); got != 0 {
		t.Errorf("absent token should score 0, got %v", got)
	}
	if got := specificity([]string{"FERRARI 488 GTB", "missing"}, extracted); got != 0.5 {
		t.Errorf("half present (case-insensitive) should score 0.5, got %v", got)
	}
	if got := specificity(nil, extracted); got != 1 {
		t.Errorf("no required tokens should score 1, got %v", got)
	}
}

// judgeFor returns a fake LLM that says YES iff both facts share a marker word.
func judgeFor(marker string) *fake.LLM {
	return &fake.LLM{Responder: func(system, user string) (string, error) {
		if strings.Count(strings.ToLower(user), strings.ToLower(marker)) >= 2 {
			return "YES", nil
		}
		return "NO", nil
	}}
}

func TestRecallPrecisionWithJudge(t *testing.T) {
	j := Judge{LLM: judgeFor("shopify")}
	ctx := context.Background()
	expected := []string{"User works at Shopify", "User owns a dog"}
	extracted := []string{"The user is employed at Shopify"}

	// One of two expected facts matched -> recall 0.5.
	if got := recall(ctx, j, expected, extracted); got != 0.5 {
		t.Errorf("recall = %v, want 0.5", got)
	}
	// The single extracted fact matches an expected -> precision 1.0.
	if got := precision(ctx, j, expected, extracted); got != 1 {
		t.Errorf("precision = %v, want 1.0", got)
	}
}

func TestJudgeExactMatchShortCircuits(t *testing.T) {
	// A nil LLM would panic if called; exact match must not call it.
	j := Judge{LLM: nil}
	if !j.Same(context.Background(), "same text", "  same text  ") {
		t.Error("identical text should match without calling the LLM")
	}
}

func TestPrecisionEmptyExtraction(t *testing.T) {
	j := Judge{LLM: judgeFor("x")}
	if got := precision(context.Background(), j, []string{"a"}, nil); got != 1 {
		t.Errorf("no extracted facts -> no false positives -> precision 1, got %v", got)
	}
}

// TestRunOffline exercises the full Run orchestration with fakes, proving the
// harness wiring works without network. A scripted LLM returns canned
// extractions; a marker-word judge scores them.
func TestRunOffline(t *testing.T) {
	st, err := sqlite.Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	llm := &fake.LLM{Responder: func(system, user string) (string, error) {
		// Judge calls contain "Statement A:"; extraction calls do not.
		if strings.Contains(user, "Statement A:") {
			if strings.Contains(user, "Shopify") {
				return "YES", nil
			}
			return "NO", nil
		}
		if strings.Contains(user, "Shopify") {
			return fake.JSON("The user works at Shopify"), nil
		}
		return `{"memory":[]}`, nil
	}}

	fixtures := []Fixture{{
		Name:          "work",
		Messages:      msg("user", "I work at Shopify"),
		ExpectedFacts: []string{"The user works at Shopify"},
	}}

	reports, err := Run(context.Background(), fixtures, Config{
		LLM:      llm,
		Embedder: &fake.Embedder{D: 64},
		Store:    st,
		Versions: []string{"v1"},
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(reports) != 1 {
		t.Fatalf("want 1 report, got %d", len(reports))
	}
	r := reports[0]
	if r.MeanRecall() != 1 {
		t.Errorf("recall = %v, want 1", r.MeanRecall())
	}
	// Re-ingesting the same fact should add nothing new (hash + existing-memory dedup).
	if r.MeanDedup() != 1 {
		t.Errorf("dedup score = %v, want 1 (re-add should write 0 new facts)", r.MeanDedup())
	}
}
