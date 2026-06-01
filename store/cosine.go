package store

import "math"

// Cosine returns the cosine similarity of two equal-length float32 vectors.
// It returns 0 when either vector is zero-length, has a zero magnitude, or the
// lengths differ — callers treat 0 as "no similarity" rather than an error.
func Cosine(a, b []float32) float32 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		dot += x * y
		na += x * x
		nb += y * y
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return float32(dot / (math.Sqrt(na) * math.Sqrt(nb)))
}
