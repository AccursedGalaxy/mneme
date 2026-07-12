// Package bench is mneme's benchmark harness for the public long-term-memory
// datasets (LoCoMo, LongMemEval). It is separate from package eval on purpose:
// eval scores extraction against our hand-authored fixtures; bench scores
// end-to-end QA over multi-session dialogues with a retrieve→answer→judge loop,
// the protocol the published benchmarks use.
//
// Like eval, the orchestration is unit-tested offline with fakes, but a real run
// needs the dataset files (gitignored, path via flag) and a live model — drive
// it via cmd/bench. The datasets are NOT downloaded by `go test`.
//
// PLAN-v2.md §4.1 is the spec; this is task #1, the v2 baseline every later task
// is measured against.
package bench

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/AccursedGalaxy/mneme/types"
)

// Sample is one benchmark case: a multi-session dialogue plus its QA pairs. It
// is dataset-agnostic — both loaders project their native schema onto this.
type Sample struct {
	ID        string    // stable id; used as the per-sample Scope.RunID
	Sessions  []Session // ingested in order, each via one Add
	Questions []QAPair
}

// Session is one block of a dialogue ingested together. Date is the raw
// human-readable timestamp from the dataset (may be empty); the runner surfaces
// it to the extractor so dates in the text have an anchor. Structured temporal
// grounding is a later task (PLAN-v2.md §5.1); here Date is informational only.
type Session struct {
	Date     string
	Messages []types.Message
}

// QAPair is one question with its gold answer and benchmark category. Category
// is the normalized label (e.g. "single_hop", "temporal") used for the
// per-category aggregation that tells us which lever to pull next.
//
// Unanswerable marks a question whose correct behavior is abstention (LoCoMo's
// adversarial category): Answer is empty and the system scores 1 only when it
// declines to answer. Its distractor answer is never used as gold — scoring
// against it would invert the metric, rewarding the hallucination the category
// exists to catch.
type QAPair struct {
	Question     string
	Answer       string
	Category     string
	Unanswerable bool
}

// flexString unmarshals a JSON value that may be a string, number, or boolean
// into a string. LoCoMo answers are sometimes bare numbers (e.g. a count) rather
// than quoted strings, which would break a plain `string` field.
type flexString string

func (f *flexString) UnmarshalJSON(b []byte) error {
	b = bytes.TrimSpace(b)
	if len(b) == 0 || string(b) == "null" {
		*f = ""
		return nil
	}
	if b[0] == '"' {
		var s string
		if err := json.Unmarshal(b, &s); err != nil {
			return err
		}
		*f = flexString(s)
		return nil
	}
	// Number / bool / other scalar: take the literal text.
	*f = flexString(string(b))
	return nil
}

// ---------------------------------------------------------------------------
// LoCoMo
// ---------------------------------------------------------------------------

// LoCoMo's category integers, per the dataset paper (Maharana et al., 2024).
// The per-category split is the whole point of the benchmark — a single
// aggregate hides which capability is weak.
func locomoCategory(c int) string {
	switch c {
	case 1:
		return "multi_hop"
	case 2:
		return "temporal"
	case 3:
		return "open_domain"
	case 4:
		return "single_hop"
	case 5:
		return "adversarial"
	default:
		return "category_" + strconv.Itoa(c)
	}
}

type locomoFile []locomoSample

type locomoSample struct {
	SampleID     string                     `json:"sample_id"`
	Conversation map[string]json.RawMessage `json:"conversation"`
	QA           []locomoQA                 `json:"qa"`
}

type locomoQA struct {
	Question          string     `json:"question"`
	Answer            flexString `json:"answer"`
	AdversarialAnswer flexString `json:"adversarial_answer"`
	Category          int        `json:"category"`
}

type locomoTurn struct {
	Speaker     string `json:"speaker"`
	Text        string `json:"text"`
	BlipCaption string `json:"blip_caption"`
}

var locomoSessionKey = regexp.MustCompile(`^session_(\d+)$`)

// LoadLoCoMo reads a LoCoMo dataset file (the public locomo10.json shape: an
// array of samples, each with a `conversation` object holding `session_N` turn
// arrays + `session_N_date_time` strings, and a `qa` array).
func LoadLoCoMo(path string) ([]Sample, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var file locomoFile
	if err := json.Unmarshal(data, &file); err != nil {
		return nil, fmt.Errorf("parse LoCoMo %s: %w", path, err)
	}

	samples := make([]Sample, 0, len(file))
	for i, ls := range file {
		id := ls.SampleID
		if id == "" {
			id = "sample_" + strconv.Itoa(i)
		}

		sessions, err := locomoSessions(ls.Conversation)
		if err != nil {
			return nil, fmt.Errorf("sample %s: %w", id, err)
		}

		var questions []QAPair
		for _, q := range ls.QA {
			if strings.TrimSpace(q.Question) == "" {
				continue
			}
			// Adversarial questions (category 5) are unanswerable by design:
			// their gold behavior is abstention, and adversarial_answer is the
			// trap (text present in the conversation but misattributed by the
			// question) — it is deliberately NOT used as a gold answer. A
			// non-adversarial question with no gold answer cannot be scored at
			// all and is dropped.
			adversarial := q.Category == 5
			gold := string(q.Answer)
			if !adversarial && gold == "" {
				continue
			}
			questions = append(questions, QAPair{
				Question:     q.Question,
				Answer:       gold,
				Category:     locomoCategory(q.Category),
				Unanswerable: adversarial,
			})
		}

		samples = append(samples, Sample{ID: id, Sessions: sessions, Questions: questions})
	}
	return samples, nil
}

// locomoSessions extracts the session_N turn arrays in numeric order, pairing
// each with its session_N_date_time. JSON object key order is undefined, so we
// discover the session numbers and sort them.
func locomoSessions(conv map[string]json.RawMessage) ([]Session, error) {
	var nums []int
	for key := range conv {
		if m := locomoSessionKey.FindStringSubmatch(key); m != nil {
			n, _ := strconv.Atoi(m[1])
			nums = append(nums, n)
		}
	}
	sort.Ints(nums)

	sessions := make([]Session, 0, len(nums))
	for _, n := range nums {
		key := "session_" + strconv.Itoa(n)
		var turns []locomoTurn
		if err := json.Unmarshal(conv[key], &turns); err != nil {
			return nil, fmt.Errorf("parse %s: %w", key, err)
		}

		var date string
		if raw, ok := conv[key+"_date_time"]; ok {
			_ = json.Unmarshal(raw, &date) // best-effort; absent/odd date is fine
		}

		msgs := make([]types.Message, 0, len(turns))
		for _, t := range turns {
			content := strings.TrimSpace(t.Text)
			if content == "" && t.BlipCaption != "" {
				content = "[shared an image: " + strings.TrimSpace(t.BlipCaption) + "]"
			}
			if content == "" {
				continue
			}
			// Both LoCoMo speakers are human participants; tag the speaker by
			// Name so the extractor can attribute facts to the right person.
			msgs = append(msgs, types.Message{Role: "user", Content: content, Name: t.Speaker})
		}
		if len(msgs) == 0 {
			continue
		}
		sessions = append(sessions, Session{Date: date, Messages: msgs})
	}
	return sessions, nil
}

// ---------------------------------------------------------------------------
// LongMemEval
// ---------------------------------------------------------------------------

type lmeFile []lmeEntry

type lmeEntry struct {
	QuestionID       string      `json:"question_id"`
	QuestionType     string      `json:"question_type"`
	Question         string      `json:"question"`
	Answer           flexString  `json:"answer"`
	HaystackDates    []string    `json:"haystack_dates"`
	HaystackSessions [][]lmeTurn `json:"haystack_sessions"`
}

type lmeTurn struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// LoadLongMemEval reads a LongMemEval dataset file (longmemeval_s.json /
// longmemeval_m.json shape: an array of entries, each with `haystack_sessions`
// — a list of sessions, each a list of {role, content} turns — plus a single
// question/answer and a `question_type` we use as the category label).
func LoadLongMemEval(path string) ([]Sample, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var file lmeFile
	if err := json.Unmarshal(data, &file); err != nil {
		return nil, fmt.Errorf("parse LongMemEval %s: %w", path, err)
	}

	samples := make([]Sample, 0, len(file))
	for i, e := range file {
		id := e.QuestionID
		if id == "" {
			id = "q_" + strconv.Itoa(i)
		}

		sessions := make([]Session, 0, len(e.HaystackSessions))
		for si, turns := range e.HaystackSessions {
			msgs := make([]types.Message, 0, len(turns))
			for _, t := range turns {
				content := strings.TrimSpace(t.Content)
				if content == "" {
					continue
				}
				role := t.Role
				if role == "" {
					role = "user"
				}
				msgs = append(msgs, types.Message{Role: role, Content: content})
			}
			if len(msgs) == 0 {
				continue
			}
			var date string
			if si < len(e.HaystackDates) {
				date = e.HaystackDates[si]
			}
			sessions = append(sessions, Session{Date: date, Messages: msgs})
		}

		samples = append(samples, Sample{
			ID:       id,
			Sessions: sessions,
			Questions: []QAPair{{
				Question: e.Question,
				Answer:   string(e.Answer),
				Category: normalizeCategory(e.QuestionType),
			}},
		})
	}
	return samples, nil
}

// normalizeCategory turns a dataset's native category label into the
// underscore form used across the report (e.g. "single-session-user" ->
// "single_session_user").
func normalizeCategory(s string) string {
	s = strings.TrimSpace(strings.ToLower(s))
	if s == "" {
		return "uncategorized"
	}
	return strings.ReplaceAll(s, "-", "_")
}
