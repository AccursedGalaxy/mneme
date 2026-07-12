package bench

import (
	"strings"
	"testing"
)

// The source oracle is only meaningful if evidence ids actually resolve to turns.
// If the wiring silently produced an empty map, the oracle would answer from
// nothing, score near zero, and read as "the dataset is unanswerable" — a
// spectacular false finding. Load the real dataset and check the join holds.
func TestLoCoMoEvidenceResolvesToTurns(t *testing.T) {
	samples, err := LoadLoCoMo("data/locomo10.json")
	if err != nil {
		t.Skipf("dataset not present (it is gitignored): %v", err)
	}

	var withEvidence, resolved int
	for _, s := range samples {
		if len(s.Turns) == 0 {
			t.Fatalf("sample %s indexed no turns by dia_id", s.ID)
		}
		for _, q := range s.Questions {
			if q.Unanswerable {
				continue // adversarial questions cite no evidence by construction
			}
			if len(q.Evidence) == 0 {
				continue
			}
			withEvidence++
			facts := answerFromEvidence(s, q)
			if len(facts) > 0 {
				resolved++
			}
		}
	}

	if withEvidence == 0 {
		t.Fatal("no answerable question carried evidence ids — the loader dropped the evidence field")
	}
	// Every cited turn should exist in the conversation it came from.
	if got := float64(resolved) / float64(withEvidence); got < 0.99 {
		t.Errorf("only %.1f%% of evidence-bearing questions resolved to turns; the dia_id join is broken", 100*got)
	}
}

func TestOracleFactsCarryTurnText(t *testing.T) {
	s := Sample{
		ID:    "conv-1",
		Turns: map[string]string{"D1:3": "(7 May 2023) Caroline: I went to the LGBTQ support group."},
	}
	q := QAPair{Question: "When did Caroline go?", Evidence: []string{"D1:3", "D9:99"}}

	facts := answerFromEvidence(s, q)

	// The known id resolves; the unknown one is skipped rather than becoming an
	// empty fact that would dilute the answer prompt.
	if len(facts) != 1 {
		t.Fatalf("got %d facts, want 1 (unknown ids must be skipped)", len(facts))
	}
	if !strings.Contains(facts[0].Text, "support group") {
		t.Errorf("fact lost the turn text: %q", facts[0].Text)
	}
}
