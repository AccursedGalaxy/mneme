package store

import (
	"math"
	"testing"
)

func TestCosine(t *testing.T) {
	tests := []struct {
		name string
		a, b []float32
		want float32
	}{
		{"identical", []float32{1, 2, 3}, []float32{1, 2, 3}, 1},
		{"opposite", []float32{1, 0}, []float32{-1, 0}, -1},
		{"orthogonal", []float32{1, 0}, []float32{0, 1}, 0},
		{"scaled is identical", []float32{1, 2, 3}, []float32{2, 4, 6}, 1},
		{"empty", nil, nil, 0},
		{"length mismatch", []float32{1, 2}, []float32{1, 2, 3}, 0},
		{"zero magnitude", []float32{0, 0}, []float32{1, 1}, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Cosine(tt.a, tt.b)
			if math.Abs(float64(got-tt.want)) > 1e-6 {
				t.Errorf("Cosine(%v, %v) = %v, want %v", tt.a, tt.b, got, tt.want)
			}
		})
	}
}

func TestCosineNaNCollapsesToZero(t *testing.T) {
	nan := float32(math.NaN())
	if got := Cosine([]float32{nan, 1}, []float32{1, 1}); got != 0 {
		t.Errorf("NaN component must yield 0, got %v", got)
	}
	if got := Cosine([]float32{1, 1}, []float32{nan, nan}); got != 0 {
		t.Errorf("NaN vector must yield 0, got %v", got)
	}
}
