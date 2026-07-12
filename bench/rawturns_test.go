package bench

import (
	"strings"
	"testing"

	"github.com/AccursedGalaxy/mneme/types"
)

func TestRenderTurnsCarriesDateAndSpeaker(t *testing.T) {
	sess := Session{
		Date: "7 May 2023",
		Messages: []types.Message{
			{Role: "user", Name: "Caroline", Content: "I finally started therapy."},
			{Role: "user", Name: "Melanie", Content: "That's a big step."},
		},
	}

	got := renderTurns(sess, sess.Messages)

	// LoCoMo asks "when did X happen" and "who said Y". A chunk that drops the
	// date or the speaker cannot answer either, however good the embedder is.
	for _, want := range []string{"7 May 2023", "Caroline:", "Melanie:", "started therapy"} {
		if !strings.Contains(got, want) {
			t.Errorf("chunk is missing %q:\n%s", want, got)
		}
	}
}

func TestRenderTurnsSkipsEmptyWindow(t *testing.T) {
	sess := Session{Date: "1 Jan 2024", Messages: []types.Message{{Role: "user", Content: "   "}}}
	if got := renderTurns(sess, sess.Messages); got != "" {
		t.Errorf("a window of blank turns rendered %q, want empty (it must not be stored)", got)
	}
}
