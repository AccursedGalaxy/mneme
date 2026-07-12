package bench

import (
	"encoding/json"
	"fmt"
	"io"
)

// Prediction is one question's full record: what was asked, what the store
// retrieved, what the answer model said, and how the harness scored it.
//
// This is the durable artifact of a run. The markdown report is an aggregate —
// once written, a scoring bug can only be fixed by paying for the whole run
// again (which is exactly what the 2026-07-12 adversarial-scoring erratum cost
// us). A dump of every prediction makes any later scoring change a local
// re-score of a JSONL file: the LLM outputs are the expensive part, and they
// are all here.
type Prediction struct {
	SampleID     string   `json:"sample_id"`
	Category     string   `json:"category"`
	Question     string   `json:"question"`
	Gold         string   `json:"gold"`         // empty when Unanswerable
	Unanswerable bool     `json:"unanswerable"` // gold behavior is abstention
	Predicted    string   `json:"predicted"`
	Retrieved    []string `json:"retrieved"` // top-k facts the answer model saw

	// Scores as this run's harness computed them. A re-score recomputes these
	// from Predicted/Gold; they are recorded so a re-score can be diffed
	// against the numbers a given run actually published.
	EM    bool    `json:"em"`
	F1    float64 `json:"f1"`
	Judge bool    `json:"judge"`
}

// WritePredictions writes one JSON object per question, newline-delimited.
// Judge is the one field a re-score cannot recompute offline (it needs the
// oracle LLM); everything else in the record is enough to recompute EM, F1 and
// the abstention rule for free.
func WritePredictions(w io.Writer, report Report) error {
	enc := json.NewEncoder(w)
	for _, r := range report.Results {
		p := Prediction{
			SampleID:     r.SampleID,
			Category:     r.Category,
			Question:     r.Question,
			Gold:         r.Gold,
			Unanswerable: r.Unanswerable,
			Predicted:    r.Predicted,
			Retrieved:    r.Retrieved,
			EM:           r.EM,
			F1:           r.F1,
			Judge:        r.Judge,
		}
		if err := enc.Encode(p); err != nil {
			return fmt.Errorf("encode prediction: %w", err)
		}
	}
	return nil
}
