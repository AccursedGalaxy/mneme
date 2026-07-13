package bench

import (
	"strings"
	"testing"
	"time"

	"github.com/AccursedGalaxy/mneme"
)

func TestParseSessionDate(t *testing.T) {
	got, ok := parseSessionDate("1:56 pm on 8 May, 2023")
	if !ok {
		t.Fatal("LoCoMo's session date format must parse; it is every fact's ObservedAt")
	}
	want := time.Date(2023, 5, 8, 13, 56, 0, 0, time.UTC)
	if !got.Equal(want) {
		t.Errorf("got %v, want %v", got, want)
	}

	// A date we cannot parse must stay zero ("unknown"). Defaulting it would
	// stamp every fact in the session with a confidently wrong timestamp, which
	// is worse than carrying none: nothing downstream could detect it.
	if got, ok := parseSessionDate("sometime last spring"); ok || !got.IsZero() {
		t.Errorf("an unparseable date must be zero/false, got %v/%v", got, ok)
	}
}

func TestIngestMessagesStampsEveryTurn(t *testing.T) {
	sess := Session{
		Date: "1:56 pm on 8 May, 2023",
		Messages: []mneme.Message{
			{Role: "user", Name: "Caroline", Content: "I went to a support group yesterday."},
			{Role: "user", Name: "Melanie", Content: "That's great!"},
		},
	}
	msgs := ingestMessages(sess)
	if len(msgs) != 3 {
		t.Fatalf("want the dated note + 2 turns, got %d", len(msgs))
	}
	want := time.Date(2023, 5, 8, 13, 56, 0, 0, time.UTC)
	for i, m := range msgs {
		if !m.Timestamp.Equal(want) {
			t.Errorf("message %d timestamp = %v, want %v", i, m.Timestamp, want)
		}
	}
}

// The whole point of ObservedAt is that it reaches the answer model. A fact that
// knows when it was said must say so, or answer prompt v2's relative-date rule
// has nothing to resolve against and the store is no better off than before.
func TestBuildAnswerUserRendersObservedAt(t *testing.T) {
	at := time.Date(2023, 5, 8, 13, 56, 0, 0, time.UTC)
	got := buildAnswerUser("When?", []mneme.Fact{
		{Text: "Caroline attended a support group.", ObservedAt: at},
		{Text: "Melanie paints."}, // no timestamp: renders bare, as before
	})
	if !strings.Contains(got, "(said on 8 May, 2023) Caroline attended a support group.") {
		t.Errorf("a dated fact must render its date, got:\n%s", got)
	}
	if !strings.Contains(got, "- Melanie paints.\n") {
		t.Errorf("an undated fact must render bare, got:\n%s", got)
	}
}

// A session date the layouts cannot parse must not stamp the zero time over
// timestamps the messages already carry. Overwriting a real timestamp with
// "unknown" is the same silent corruption the mechanism exists to prevent.
func TestIngestMessagesKeepsExistingTimestampsWhenDateUnparseable(t *testing.T) {
	own := time.Date(2023, 5, 8, 13, 56, 0, 0, time.UTC)
	sess := Session{
		Date: "sometime last spring", // matches no layout
		Messages: []mneme.Message{
			{Role: "user", Content: "I went to a support group yesterday.", Timestamp: own},
		},
	}
	msgs := ingestMessages(sess)
	if len(msgs) != 2 {
		t.Fatalf("want the prose note + 1 turn, got %d", len(msgs))
	}
	if !msgs[1].Timestamp.Equal(own) {
		t.Errorf("an unparseable session date must leave the message's own timestamp alone, got %v want %v",
			msgs[1].Timestamp, own)
	}
}

// An unparsed session date is invisible in the score — it just reads as "temporal
// did not improve" — so the harness must count it and be able to say so.
func TestUnparsedSessionDatesIsCounted(t *testing.T) {
	before := UnparsedSessionDates()
	ingestMessages(Session{
		Date:     "sometime last spring",
		Messages: []mneme.Message{{Role: "user", Content: "hi"}},
	})
	if got := UnparsedSessionDates(); got != before+1 {
		t.Errorf("an unparseable session date must be counted: %d -> %d", before, got)
	}
}

// LongMemEval's published haystack_dates carry a weekday and time; the ISO-only
// assumption would have silently dropped every one of them.
func TestParseSessionDateCoversLongMemEvalShapes(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want time.Time
	}{
		{"2023/05/20 (Sat) 02:21", time.Date(2023, 5, 20, 2, 21, 0, 0, time.UTC)},
		{"2023-05-08", time.Date(2023, 5, 8, 0, 0, 0, 0, time.UTC)},
	} {
		got, ok := parseSessionDate(tc.in)
		if !ok || !got.Equal(tc.want) {
			t.Errorf("parseSessionDate(%q) = %v/%v, want %v/true", tc.in, got, ok, tc.want)
		}
	}
}
