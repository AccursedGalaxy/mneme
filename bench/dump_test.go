package bench

import (
	"encoding/json"
	"strings"
	"testing"
)

// A dumped run must be re-scorable offline: every field the scorer reads has to
// survive the round trip, including the unanswerable flag that carries the
// abstention rule for adversarial questions.
func TestWritePredictionsRoundTrip(t *testing.T) {
	report := Report{Results: []QAResult{
		{
			SampleID:  "conv-1",
			Category:  "single_hop",
			Question:  "What pet does Ana have?",
			Gold:      "a corgi",
			Predicted: "Ana has a corgi.",
			Retrieved: []string{"Ana adopted a corgi in May 2023."},
			EM:        false,
			F1:        0.5,
			Judge:     true,
		},
		{
			SampleID:     "conv-1",
			Category:     "adversarial",
			Question:     "What color is Ana's boat?",
			Unanswerable: true,
			Predicted:    "I don't know.",
			EM:           true,
			F1:           1,
			Judge:        true,
		},
	}}

	var buf strings.Builder
	if err := WritePredictions(&buf, report); err != nil {
		t.Fatalf("WritePredictions: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(lines) != 2 {
		t.Fatalf("got %d lines, want 2 (one JSON object per question)", len(lines))
	}

	var got []Prediction
	for i, line := range lines {
		var p Prediction
		if err := json.Unmarshal([]byte(line), &p); err != nil {
			t.Fatalf("line %d is not valid JSON: %v", i, err)
		}
		got = append(got, p)
	}

	if got[0].Gold != "a corgi" || got[0].Predicted != "Ana has a corgi." || got[0].F1 != 0.5 {
		t.Errorf("answerable record lost data: %+v", got[0])
	}
	if len(got[0].Retrieved) != 1 {
		t.Errorf("retrieved facts not preserved: %+v", got[0].Retrieved)
	}

	adv := got[1]
	if !adv.Unanswerable {
		t.Error("adversarial record lost the unanswerable flag — a re-score would grade the abstention against an empty gold")
	}
	if adv.Gold != "" {
		t.Errorf("adversarial gold = %q, want empty (the trap answer must never be dumped as gold)", adv.Gold)
	}
	if adv.Predicted != "I don't know." {
		t.Errorf("adversarial prediction = %q, want the abstention preserved verbatim", adv.Predicted)
	}
}
