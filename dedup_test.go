package mneme

import "testing"

func TestHashTextStableAndTrimmed(t *testing.T) {
	if hashText("hello") != hashText("hello") {
		t.Error("hash not stable")
	}
	if hashText("  hello  ") != hashText("hello") {
		t.Error("hash should ignore surrounding whitespace")
	}
	if hashText("hello") == hashText("world") {
		t.Error("distinct texts should not collide")
	}
	if len(hashText("x")) != 32 {
		t.Errorf("md5 hex should be 32 chars, got %d", len(hashText("x")))
	}
}

func TestDedupExistingAndBatch(t *testing.T) {
	existing := map[string]struct{}{
		hashText("already known fact"): {},
	}
	facts := []extractedFact{
		{Text: "already known fact"}, // dropped: in existing
		{Text: "brand new fact"},     // kept
		{Text: "brand new fact"},     // dropped: in-batch dup
		{Text: "  brand new fact  "}, // dropped: same hash after trim
		{Text: "another new fact"},   // kept
	}
	kept, hashes := dedup(facts, existing)
	if len(kept) != 2 {
		t.Fatalf("want 2 kept, got %d: %v", len(kept), texts(kept))
	}
	if kept[0].Text != "brand new fact" || kept[1].Text != "another new fact" {
		t.Errorf("kept order/content wrong: %v", texts(kept))
	}
	if len(hashes) != 2 || hashes[0] != hashText("brand new fact") {
		t.Errorf("hashes mismatch: %v", hashes)
	}
}

func TestDedupEmpty(t *testing.T) {
	kept, hashes := dedup(nil, nil)
	if len(kept) != 0 || len(hashes) != 0 {
		t.Errorf("empty input should yield empty output")
	}
}
