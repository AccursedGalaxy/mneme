package mneme

import (
	"crypto/md5"
	"encoding/hex"
	"strings"
)

// hashText is the dedup key for a fact: md5 of its whitespace-trimmed text.
// Two facts with identical text (modulo surrounding whitespace) share a hash
// and are treated as the same memory.
func hashText(text string) string {
	sum := md5.Sum([]byte(strings.TrimSpace(text)))
	return hex.EncodeToString(sum[:])
}

// dedup filters facts down to those worth writing: it drops any fact whose hash
// already exists in scope (existing) and any in-batch duplicate. It returns the
// surviving facts in input order alongside each survivor's hash (same order),
// so the caller can persist text+hash without recomputing.
func dedup(facts []extractedFact, existing map[string]struct{}) (kept []extractedFact, hashes []string) {
	seen := make(map[string]struct{}, len(facts))
	for _, f := range facts {
		h := hashText(f.Text)
		if _, ok := existing[h]; ok {
			continue
		}
		if _, ok := seen[h]; ok {
			continue
		}
		seen[h] = struct{}{}
		kept = append(kept, f)
		hashes = append(hashes, h)
	}
	return kept, hashes
}
