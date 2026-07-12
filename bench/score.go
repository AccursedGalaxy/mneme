package bench

import (
	"sort"
	"strings"
)

// QAResult is one question's score under the harness: the deterministic
// overlap metrics (exact-match, token F1) and the LLM judge's semantic verdict.
// We report both because deterministic metrics are reproducible but brittle on
// paraphrase, while the judge tolerates wording but costs a call and can drift —
// neither alone is trustworthy.
type QAResult struct {
	SampleID  string
	Question  string
	Gold      string
	Predicted string
	Category  string

	// Unanswerable mirrors QAPair.Unanswerable: the gold behavior is abstention
	// and Gold is empty. Carried on the result so a re-score from a dumped run
	// can apply the abstention rule without re-reading the dataset.
	Unanswerable bool

	// Retrieved is the text of the top-k facts fed to the answer model for this
	// question. Carried for the -v diagnostic dump so a wrong answer can be
	// traced to a missing/merged/distorted fact rather than just observed.
	Retrieved []string

	EM    bool    // normalized exact match
	F1    float64 // SQuAD-style token-overlap F1
	Judge bool    // eval.Judge semantic match
}

// Report is the full run: every question's result plus the run parameters. The
// per-category breakdown (Categories) is the table that tells us what to fix
// next; Overall is the headline number.
type Report struct {
	Results  []QAResult
	K        int
	Strategy string
	// ConsolidationVersion records which consolidation prompt version was used
	// (empty for additive runs or the library default), for run provenance.
	ConsolidationVersion string
	// Rerank records whether a rerank pass was active in Search; MultiQuery
	// records the query-expansion fan-out (0/1 = off), both for run provenance.
	Rerank     bool
	MultiQuery int
}

// CategoryStat aggregates one category (or the whole run) into mean metrics.
type CategoryStat struct {
	Category string
	N        int
	EM       float64 // mean exact-match (fraction)
	F1       float64 // mean token F1
	Judge    float64 // mean semantic-match (fraction)
}

// Overall aggregates every result into a single CategoryStat labelled "overall".
func (r Report) Overall() CategoryStat {
	return aggregate("overall", r.Results)
}

// Categories returns one CategoryStat per category, sorted by name for a stable
// table. A single aggregate hides which capability is weak; this is the point
// of the harness.
func (r Report) Categories() []CategoryStat {
	byCat := map[string][]QAResult{}
	for _, res := range r.Results {
		byCat[res.Category] = append(byCat[res.Category], res)
	}
	cats := make([]string, 0, len(byCat))
	for c := range byCat {
		cats = append(cats, c)
	}
	sort.Strings(cats)
	out := make([]CategoryStat, 0, len(cats))
	for _, c := range cats {
		out = append(out, aggregate(c, byCat[c]))
	}
	return out
}

func aggregate(label string, rs []QAResult) CategoryStat {
	stat := CategoryStat{Category: label, N: len(rs)}
	if len(rs) == 0 {
		return stat
	}
	var em, f1, judge float64
	for _, r := range rs {
		if r.EM {
			em++
		}
		f1 += r.F1
		if r.Judge {
			judge++
		}
	}
	n := float64(len(rs))
	stat.EM = em / n
	stat.F1 = f1 / n
	stat.Judge = judge / n
	return stat
}

// exactMatch reports whether two answers match after normalization.
func exactMatch(pred, gold string) bool {
	return normalize(pred) == normalize(gold)
}

// abstentionMarkers are normalized substrings that signal the answer model
// declined to answer. The QA prompt mandates the exact phrase "I don't know.",
// but the check tolerates the near-synonyms models actually emit.
var abstentionMarkers = []string{
	"i don t know",
	"don t know",
	"not mentioned",
	"no information",
	"not specified",
	"cannot be determined",
	"unanswerable",
}

// abstained reports whether a predicted answer is an abstention. It is the
// scoring rule for unanswerable (adversarial) questions, where declining to
// answer is the gold behavior and any concrete answer is a hallucination.
func abstained(pred string) bool {
	n := normalize(pred)
	if n == "" {
		return true
	}
	for _, m := range abstentionMarkers {
		if strings.Contains(n, m) {
			return true
		}
	}
	return false
}

// f1 is the SQuAD-style token-overlap F1 between a predicted and gold answer,
// computed over normalized tokens with multiset intersection. It rewards
// partial answers (the right name buried in a sentence) where exact-match
// would score 0.
func f1(pred, gold string) float64 {
	pTok := strings.Fields(normalize(pred))
	gTok := strings.Fields(normalize(gold))
	if len(pTok) == 0 || len(gTok) == 0 {
		// If both are empty they trivially agree; otherwise no overlap.
		if len(pTok) == 0 && len(gTok) == 0 {
			return 1
		}
		return 0
	}

	goldCounts := map[string]int{}
	for _, t := range gTok {
		goldCounts[t]++
	}
	common := 0
	for _, t := range pTok {
		if goldCounts[t] > 0 {
			goldCounts[t]--
			common++
		}
	}
	if common == 0 {
		return 0
	}
	precision := float64(common) / float64(len(pTok))
	recall := float64(common) / float64(len(gTok))
	return 2 * precision * recall / (precision + recall)
}

// articles dropped during normalization, SQuAD-style, so "the dog" == "dog".
var articles = map[string]bool{"a": true, "an": true, "the": true}

// normalize lowercases, strips punctuation to spaces, drops articles, and
// collapses whitespace — the standard QA-answer normalization so EM/F1 are not
// fooled by casing, "the", or trailing punctuation.
func normalize(s string) string {
	s = strings.ToLower(s)
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
			b.WriteRune(r)
		default:
			b.WriteByte(' ')
		}
	}
	fields := strings.Fields(b.String())
	kept := fields[:0]
	for _, f := range fields {
		if !articles[f] {
			kept = append(kept, f)
		}
	}
	return strings.Join(kept, " ")
}
