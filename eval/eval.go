// Package eval is mneme's prompt evaluation harness. It runs hand-authored
// fixtures through the real Add/Search pipeline against a live LLM and scores
// extraction quality, so a prompt version is a decision backed by numbers and a
// prompt change can be caught when it regresses. It is not part of
// `go test ./...` (which must run offline); drive it via cmd/eval.
package eval

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/AccursedGalaxy/mneme/types"
)

// Query is a retrieval probe: after a fixture is ingested, searching Q must
// surface a fact semantically matching one of ShouldRecall within top-k.
type Query struct {
	Q            string   `json:"q"`
	ShouldRecall []string `json:"should_recall"`
}

// Fixture is one LoCoMo-style evaluation case.
type Fixture struct {
	Name string `json:"name"`
	// Messages is the conversation fed to Add.
	Messages []types.Message `json:"messages"`
	// ExpectedFacts are the durable facts a good extractor should produce.
	ExpectedFacts []string `json:"expected_facts"`
	// Queries probe search recall (optional).
	Queries []Query `json:"queries"`
	// RequiredTokens are substrings that must survive verbatim into some
	// extracted fact — the deterministic specificity check (optional).
	RequiredTokens []string `json:"required_tokens,omitempty"`
	// SkipDedup opts a fixture out of the dedup-correctness check (e.g. the
	// nothing-to-extract case, where there is nothing to re-dedup).
	SkipDedup bool `json:"skip_dedup,omitempty"`
}

// LoadFixtures reads and parses every *.json file in dir, sorted by name for a
// stable run order.
func LoadFixtures(dir string) ([]Fixture, error) {
	paths, err := filepath.Glob(filepath.Join(dir, "*.json"))
	if err != nil {
		return nil, err
	}
	sort.Strings(paths)
	var fixtures []Fixture
	for _, p := range paths {
		data, err := os.ReadFile(p)
		if err != nil {
			return nil, err
		}
		var f Fixture
		if err := json.Unmarshal(data, &f); err != nil {
			return nil, fmt.Errorf("parse fixture %s: %w", p, err)
		}
		if f.Name == "" {
			f.Name = filepath.Base(p)
		}
		fixtures = append(fixtures, f)
	}
	return fixtures, nil
}
