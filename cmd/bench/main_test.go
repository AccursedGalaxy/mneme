package main

import (
	"testing"

	"github.com/AccursedGalaxy/mneme/bench"
)

func qs(cats ...string) []bench.QAPair {
	out := make([]bench.QAPair, len(cats))
	for i, c := range cats {
		out[i] = bench.QAPair{Category: c, Question: c + "?"}
	}
	return out
}

func countByCat(qs []bench.QAPair) map[string]int {
	m := map[string]int{}
	for _, q := range qs {
		m[q.Category]++
	}
	return m
}

func TestCapBalanced(t *testing.T) {
	// First-N would take 3 "a"s; balanced must spread across categories.
	in := qs("a", "a", "a", "b", "b", "c")
	got := capBalanced(in, 3)
	if len(got) != 3 {
		t.Fatalf("len = %d, want 3", len(got))
	}
	c := countByCat(got)
	if c["a"] != 1 || c["b"] != 1 || c["c"] != 1 {
		t.Errorf("not balanced across categories: %v", c)
	}
}

func TestCapBalancedUneven(t *testing.T) {
	// cap larger than some categories: round-robin keeps drawing from the ones
	// that still have questions until the cap is met.
	in := qs("a", "a", "a", "a", "b")
	got := capBalanced(in, 3)
	if len(got) != 3 {
		t.Fatalf("len = %d, want 3", len(got))
	}
	c := countByCat(got)
	// Round 1: a,b -> 1 each (2). Round 2: a (b empty) -> a=2. Total 3.
	if c["a"] != 2 || c["b"] != 1 {
		t.Errorf("uneven draw = %v, want a:2 b:1", c)
	}
}

func TestCapBalancedNoOpWhenSmall(t *testing.T) {
	in := qs("a", "b")
	if got := capBalanced(in, 5); len(got) != 2 {
		t.Errorf("cap >= len should return all: got %d", len(got))
	}
	if got := capBalanced(in, 0); len(got) != 2 {
		t.Errorf("cap 0 should return all (no cap): got %d", len(got))
	}
}
