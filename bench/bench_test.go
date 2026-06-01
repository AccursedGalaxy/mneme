package bench

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/AccursedGalaxy/mneme/provider/fake"
	"github.com/AccursedGalaxy/mneme/store/sqlite"
	"github.com/AccursedGalaxy/mneme/types"
)

// turn builds a one-message slice for a session in tests.
func turn(role, content string) []types.Message {
	return []types.Message{{Role: role, Content: content}}
}

// writeTemp writes content to a temp file (name is a glob-ish base) and returns
// its path; the file is cleaned up when the test ends.
func writeTemp(t *testing.T, name, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), strings.ReplaceAll(name, "*", "x"))
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// --- deterministic scoring -------------------------------------------------

func TestNormalize(t *testing.T) {
	cases := map[string]string{
		"The Dog.":          "dog",
		"  New   York  ":    "new york",
		"$1,234!":           "1 234",
		"An apple a day":    "apple day",
		"I don't know.":     "i don t know",
		"Caroline's beagle": "caroline s beagle",
	}
	for in, want := range cases {
		if got := normalize(in); got != want {
			t.Errorf("normalize(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestExactMatch(t *testing.T) {
	if !exactMatch("The dog", "dog!") {
		t.Error("article + punctuation should not break exact match")
	}
	if exactMatch("cat", "dog") {
		t.Error("different answers must not match")
	}
}

func TestF1(t *testing.T) {
	// Full overlap (modulo normalization) -> 1.
	if got := f1("New York City", "new york city"); got != 1 {
		t.Errorf("identical (normalized) F1 = %v, want 1", got)
	}
	// No overlap -> 0.
	if got := f1("cat", "dog"); got != 0 {
		t.Errorf("disjoint F1 = %v, want 0", got)
	}
	// Partial overlap: pred has the right token buried in noise.
	// gold tokens {paris}; pred tokens {the,answer,is,paris} -> "the" dropped as
	// article, so pred {answer,is,paris}: common=1, P=1/3, R=1, F1=0.5.
	got := f1("the answer is Paris", "Paris")
	if got < 0.49 || got > 0.51 {
		t.Errorf("partial F1 = %v, want ~0.5", got)
	}
	// Both empty -> trivially agree.
	if got := f1("", ""); got != 1 {
		t.Errorf("both-empty F1 = %v, want 1", got)
	}
	// One empty -> 0.
	if got := f1("", "something"); got != 0 {
		t.Errorf("one-empty F1 = %v, want 0", got)
	}
}

// --- aggregation -----------------------------------------------------------

func TestReportAggregation(t *testing.T) {
	r := Report{Results: []QAResult{
		{Category: "single_hop", EM: true, F1: 1, Judge: true},
		{Category: "single_hop", EM: false, F1: 0, Judge: false},
		{Category: "temporal", EM: false, F1: 0.5, Judge: true},
	}}

	overall := r.Overall()
	if overall.N != 3 {
		t.Fatalf("overall N = %d, want 3", overall.N)
	}
	if overall.EM != 1.0/3 {
		t.Errorf("overall EM = %v, want 1/3", overall.EM)
	}
	wantF1 := (1.0 + 0 + 0.5) / 3
	if overall.F1 != wantF1 {
		t.Errorf("overall F1 = %v, want %v", overall.F1, wantF1)
	}

	cats := r.Categories()
	if len(cats) != 2 {
		t.Fatalf("want 2 categories, got %d", len(cats))
	}
	// Sorted by name: single_hop before temporal.
	if cats[0].Category != "single_hop" || cats[1].Category != "temporal" {
		t.Fatalf("categories not sorted: %v", cats)
	}
	if cats[0].EM != 0.5 || cats[0].Judge != 0.5 {
		t.Errorf("single_hop EM/Judge = %v/%v, want 0.5/0.5", cats[0].EM, cats[0].Judge)
	}
	if cats[1].N != 1 || cats[1].F1 != 0.5 {
		t.Errorf("temporal N/F1 = %d/%v, want 1/0.5", cats[1].N, cats[1].F1)
	}
}

// --- loaders ---------------------------------------------------------------

func TestLoadLoCoMo(t *testing.T) {
	const data = `[
	  {
	    "sample_id": "conv-1",
	    "conversation": {
	      "speaker_a": "Caroline",
	      "speaker_b": "Melanie",
	      "session_1_date_time": "1:56 pm on 8 May, 2023",
	      "session_1": [
	        {"speaker": "Caroline", "dia_id": "D1:1", "text": "I adopted a beagle named Max."},
	        {"speaker": "Melanie", "dia_id": "D1:2", "text": "", "blip_caption": "a dog in a park"}
	      ],
	      "session_2_date_time": "10:00 am on 9 May, 2023",
	      "session_2": [
	        {"speaker": "Caroline", "dia_id": "D2:1", "text": "Max is now six months old."}
	      ]
	    },
	    "qa": [
	      {"question": "What is the name of Caroline's dog?", "answer": "Max", "category": 4},
	      {"question": "How many cats does Caroline own?", "adversarial_answer": "Not mentioned.", "category": 5},
	      {"question": "How many dogs?", "answer": 1, "category": 4}
	    ]
	  }
	]`
	path := writeTemp(t, "locomo*.json", data)

	samples, err := LoadLoCoMo(path)
	if err != nil {
		t.Fatalf("LoadLoCoMo: %v", err)
	}
	if len(samples) != 1 {
		t.Fatalf("want 1 sample, got %d", len(samples))
	}
	s := samples[0]
	if s.ID != "conv-1" {
		t.Errorf("ID = %q, want conv-1", s.ID)
	}
	// Two sessions, in numeric order, with dates carried.
	if len(s.Sessions) != 2 {
		t.Fatalf("want 2 sessions, got %d", len(s.Sessions))
	}
	if s.Sessions[0].Date != "1:56 pm on 8 May, 2023" {
		t.Errorf("session 1 date = %q", s.Sessions[0].Date)
	}
	// The empty-text image turn folds in its blip_caption rather than vanishing
	// silently — so the image is at least represented.
	if len(s.Sessions[0].Messages) != 2 {
		t.Fatalf("session 1 messages = %d, want 2 (caption folded in)", len(s.Sessions[0].Messages))
	}
	if !strings.Contains(s.Sessions[0].Messages[1].Content, "a dog in a park") {
		t.Errorf("blip caption not folded in: %q", s.Sessions[0].Messages[1].Content)
	}
	if s.Sessions[0].Messages[0].Name != "Caroline" {
		t.Errorf("speaker not carried as Name: %q", s.Sessions[0].Messages[0].Name)
	}
	// Three questions; categories normalized; adversarial gold from
	// adversarial_answer; numeric answer coerced to a string.
	if len(s.Questions) != 3 {
		t.Fatalf("want 3 questions, got %d", len(s.Questions))
	}
	if s.Questions[0].Category != "single_hop" {
		t.Errorf("q0 category = %q, want single_hop", s.Questions[0].Category)
	}
	if s.Questions[1].Category != "adversarial" || s.Questions[1].Answer != "Not mentioned." {
		t.Errorf("q1 = %+v, want adversarial gold from adversarial_answer", s.Questions[1])
	}
	if s.Questions[2].Answer != "1" {
		t.Errorf("numeric answer = %q, want \"1\"", s.Questions[2].Answer)
	}
}

func TestLoadLongMemEval(t *testing.T) {
	const data = `[
	  {
	    "question_id": "q1",
	    "question_type": "single-session-user",
	    "question": "Where does the user work?",
	    "answer": "Shopify",
	    "haystack_dates": ["2023-01-01", "2023-02-01"],
	    "haystack_sessions": [
	      [{"role": "user", "content": "I just started at Shopify."},
	       {"role": "assistant", "content": "Congrats!"}],
	      [{"role": "user", "content": "Busy week."}]
	    ]
	  }
	]`
	path := writeTemp(t, "lme*.json", data)

	samples, err := LoadLongMemEval(path)
	if err != nil {
		t.Fatalf("LoadLongMemEval: %v", err)
	}
	if len(samples) != 1 {
		t.Fatalf("want 1 sample, got %d", len(samples))
	}
	s := samples[0]
	if s.ID != "q1" || len(s.Sessions) != 2 {
		t.Fatalf("unexpected sample: id=%q sessions=%d", s.ID, len(s.Sessions))
	}
	if s.Sessions[0].Date != "2023-01-01" || s.Sessions[1].Date != "2023-02-01" {
		t.Errorf("haystack dates not paired by index: %q / %q", s.Sessions[0].Date, s.Sessions[1].Date)
	}
	if len(s.Questions) != 1 || s.Questions[0].Category != "single_session_user" {
		t.Fatalf("unexpected question: %+v", s.Questions)
	}
	if s.Questions[0].Answer != "Shopify" {
		t.Errorf("answer = %q, want Shopify", s.Questions[0].Answer)
	}
}

// --- full offline run ------------------------------------------------------

// TestRunOffline exercises the whole protocol — ingest, retrieve, answer,
// score — with fakes and no network, proving the harness wiring works. A single
// scripted LLM plays three roles (extractor, answerer, judge), branching on
// markers unique to each prompt.
func TestRunOffline(t *testing.T) {
	st, err := sqlite.Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	llm := &fake.LLM{Responder: func(system, user string) (string, error) {
		switch {
		case strings.Contains(user, "Statement A:"): // judge (eval.Judge)
			// Match when both statements mention the same key token.
			if mentionsBoth(user, "shopify") || mentionsBoth(user, "max") {
				return "YES", nil
			}
			return "NO", nil

		case strings.Contains(system, "answer a QUESTION"): // answerer
			// Branch on the QUESTION, not the MEMORIES: with only two facts in
			// scope and k=5, both facts are retrieved for every question, so
			// keying off memory contents would cross the wires.
			q := user[strings.Index(user, "QUESTION:"):]
			switch {
			case strings.Contains(q, "work") && strings.Contains(user, "Shopify"):
				return "Shopify", nil
			case strings.Contains(q, "dog") && strings.Contains(user, "Max"):
				return "Max", nil
			}
			return "I don't know.", nil

		default: // extractor
			// Read only the NEW MESSAGES block: existing memories are also in the
			// prompt (for dedup), so keying off the whole prompt would re-extract
			// a prior fact on the next session.
			nm := user[strings.Index(user, "NEW MESSAGES"):]
			switch {
			case strings.Contains(nm, "Shopify"):
				return fake.JSON("The user works at Shopify"), nil
			case strings.Contains(nm, "Max"):
				return fake.JSON("Caroline's dog is named Max"), nil
			}
			return `{"memory":[]}`, nil
		}
	}}

	samples := []Sample{{
		ID: "s1",
		Sessions: []Session{
			{Date: "2023-01-01", Messages: turn("user", "I work at Shopify")},
			{Messages: turn("user", "My dog Max is a beagle")},
		},
		Questions: []QAPair{
			{Question: "Where does the user work?", Answer: "Shopify", Category: "single_hop"},
			{Question: "What is the dog's name?", Answer: "Max", Category: "single_hop"},
		},
	}}

	var lastDone, lastTotal int
	report, err := Run(context.Background(), samples, Config{
		LLM:      llm,
		Embedder: &fake.Embedder{D: 64},
		Store:    st,
		K:        5,
		Progress: func(done, total int) { lastDone, lastTotal = done, total },
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	if len(report.Results) != 2 {
		t.Fatalf("want 2 results, got %d", len(report.Results))
	}
	overall := report.Overall()
	if overall.EM != 1 {
		t.Errorf("EM = %v, want 1 (both answered exactly)", overall.EM)
	}
	if overall.F1 != 1 {
		t.Errorf("F1 = %v, want 1", overall.F1)
	}
	if overall.Judge != 1 {
		t.Errorf("Judge = %v, want 1", overall.Judge)
	}
	if lastDone != 1 || lastTotal != 1 {
		t.Errorf("progress callback = %d/%d, want 1/1", lastDone, lastTotal)
	}
}

// TestRunConsolidateStrategy runs the harness end-to-end under the consolidate
// strategy: a sample whose two sessions change a fact ("Seattle" -> "Austin")
// must end with the updated answer, proving the strategy is wired through Run.
func TestRunConsolidateStrategy(t *testing.T) {
	st, err := sqlite.Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	llm := &fake.LLM{Responder: func(system, user string) (string, error) {
		switch {
		case strings.Contains(user, "Statement A:"): // judge
			if mentionsBoth(user, "austin") {
				return "YES", nil
			}
			return "NO", nil

		case strings.Contains(system, "answer a QUESTION"): // answerer
			if strings.Contains(user, "Austin") {
				return "Austin", nil
			}
			if strings.Contains(user, "Seattle") {
				return "Seattle", nil
			}
			return "I don't know.", nil

		case strings.Contains(system, "maintain a person's long-term memory"): // consolidation
			return `{"memory":[{"id":"0","text":"The user lives in Austin","event":"UPDATE"}]}`, nil

		default: // extractor (read only NEW MESSAGES)
			nm := user[strings.Index(user, "NEW MESSAGES"):]
			if strings.Contains(nm, "Austin") {
				return fake.JSON("The user lives in Austin"), nil
			}
			return fake.JSON("The user lives in Seattle"), nil
		}
	}}

	samples := []Sample{{
		ID: "move",
		Sessions: []Session{
			{Messages: turn("user", "I live in Seattle")},
			{Messages: turn("user", "I moved to Austin")},
		},
		Questions: []QAPair{{Question: "Where does the user live?", Answer: "Austin", Category: "temporal"}},
	}}

	report, err := Run(context.Background(), samples, Config{
		LLM:      llm,
		Embedder: &fake.Embedder{D: 64},
		Store:    st,
		Strategy: StrategyConsolidate,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if report.Strategy != StrategyConsolidate {
		t.Errorf("report strategy = %q, want consolidate", report.Strategy)
	}
	if len(report.Results) != 1 || !report.Results[0].EM {
		t.Errorf("consolidate run should answer the updated fact exactly: %+v", report.Results)
	}
}

func TestRunUnknownStrategy(t *testing.T) {
	st, _ := sqlite.Open(":memory:")
	defer st.Close()
	_, err := Run(context.Background(), nil, Config{
		LLM:      &fake.LLM{Default: `{"memory":[]}`},
		Embedder: &fake.Embedder{D: 64},
		Store:    st,
		Strategy: "bogus",
	})
	if err == nil || !strings.Contains(err.Error(), "unknown strategy") {
		t.Errorf("want an unknown-strategy error, got %v", err)
	}
}

// mentionsBoth reports whether token appears at least twice in s (used by the
// fake judge: both Statement A and Statement B mention it).
func mentionsBoth(s, token string) bool {
	return strings.Count(strings.ToLower(s), token) >= 2
}
