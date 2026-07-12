package bench

import "testing"

func TestAbstained(t *testing.T) {
	yes := []string{
		"I don't know.",
		"i don't know",
		"That is not mentioned in the conversation.",
		"There is no information about this.",
		"Not specified.",
		"The answer cannot be determined from the memories.",
		"",
	}
	for _, p := range yes {
		if !abstained(p) {
			t.Errorf("abstained(%q) = false, want true", p)
		}
	}
	no := []string{
		"Caroline owns 3 cats",
		"self-care is important",
		"8 May 2023",
	}
	for _, p := range no {
		if abstained(p) {
			t.Errorf("abstained(%q) = true, want false", p)
		}
	}
}
