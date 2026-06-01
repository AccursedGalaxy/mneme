package mneme

import (
	"encoding/json"
	"strings"
)

// extractedFact is one item the extractor LLM returns. IDs are the small
// integer-strings ("0","1",...) the prompt asks for; they are mapped back to
// real UUIDs / used only for the LLM's own referencing, never stored.
type extractedFact struct {
	ID           string `json:"id"`
	Text         string `json:"text"`
	AttributedTo string `json:"attributed_to"`
}

// extractionEnvelope is the documented response shape:
// {"memory":[{"id","text","attributed_to"}]}.
type extractionEnvelope struct {
	Memory []extractedFact `json:"memory"`
}

// parseExtraction pulls the fact list out of a raw LLM response defensively.
// It tolerates markdown code fences, a JSON object embedded in prose, and a
// bare array fallback. On any unrecoverable parse failure it returns nil —
// "nothing extracted" — rather than an error, so a malformed model response
// can never panic or fail the whole Add.
func parseExtraction(raw string) []extractedFact {
	s := stripFences(raw)

	// Fast path: the whole (stripped) string is the envelope.
	if facts, ok := tryEnvelope(s); ok {
		return clean(facts)
	}

	// Find the outermost {...} or [...] embedded in surrounding prose.
	if obj := firstBalanced(s, '{', '}'); obj != "" {
		if facts, ok := tryEnvelope(obj); ok {
			return clean(facts)
		}
	}
	if arr := firstBalanced(s, '[', ']'); arr != "" {
		var facts []extractedFact
		if err := json.Unmarshal([]byte(arr), &facts); err == nil {
			return clean(facts)
		}
	}
	return nil
}

func tryEnvelope(s string) ([]extractedFact, bool) {
	var env extractionEnvelope
	if err := json.Unmarshal([]byte(s), &env); err == nil && env.Memory != nil {
		return env.Memory, true
	}
	return nil, false
}

// clean drops items with empty text and trims surrounding whitespace, so a
// model that emits {"text":""} or stray spaces does not create junk facts.
func clean(facts []extractedFact) []extractedFact {
	var out []extractedFact
	for _, f := range facts {
		f.Text = strings.TrimSpace(f.Text)
		if f.Text == "" {
			continue
		}
		out = append(out, f)
	}
	return out
}

// stripFences removes a leading/trailing markdown code fence (```json ... ```)
// if present, returning the inner content trimmed.
func stripFences(s string) string {
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, "```") {
		return s
	}
	s = strings.TrimPrefix(s, "```")
	// Drop an optional language tag on the first line (e.g. "json").
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		first := strings.TrimSpace(s[:i])
		if !strings.ContainsAny(first, "{}[]") {
			s = s[i+1:]
		}
	}
	if i := strings.LastIndex(s, "```"); i >= 0 {
		s = s[:i]
	}
	return strings.TrimSpace(s)
}

// firstBalanced returns the substring from the first `open` to its matching
// `close`, accounting for nesting and ignoring braces inside JSON strings.
// Returns "" if no balanced span is found.
func firstBalanced(s string, open, close byte) string {
	start := strings.IndexByte(s, open)
	if start < 0 {
		return ""
	}
	depth := 0
	inStr := false
	escaped := false
	for i := start; i < len(s); i++ {
		c := s[i]
		if inStr {
			switch {
			case escaped:
				escaped = false
			case c == '\\':
				escaped = true
			case c == '"':
				inStr = false
			}
			continue
		}
		switch c {
		case '"':
			inStr = true
		case open:
			depth++
		case close:
			depth--
			if depth == 0 {
				return s[start : i+1]
			}
		}
	}
	return ""
}
