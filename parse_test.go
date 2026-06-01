package mneme

import "testing"

func TestParseExtraction(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want []string // expected texts in order
	}{
		{
			name: "clean object",
			raw:  `{"memory":[{"id":"0","text":"Alice drives a Ferrari","attributed_to":"user"}]}`,
			want: []string{"Alice drives a Ferrari"},
		},
		{
			name: "fenced json",
			raw:  "```json\n{\"memory\":[{\"id\":\"0\",\"text\":\"fact one\"}]}\n```",
			want: []string{"fact one"},
		},
		{
			name: "fenced no language tag",
			raw:  "```\n{\"memory\":[{\"id\":\"0\",\"text\":\"fact\"}]}\n```",
			want: []string{"fact"},
		},
		{
			name: "prose wrapped",
			raw:  `Sure! Here are the facts: {"memory":[{"id":"0","text":"prose fact"}]} Let me know if you need more.`,
			want: []string{"prose fact"},
		},
		{
			name: "empty memory list",
			raw:  `{"memory":[]}`,
			want: nil,
		},
		{
			name: "bare array fallback",
			raw:  `[{"id":"0","text":"array fact"}]`,
			want: []string{"array fact"},
		},
		{
			name: "drops empty text",
			raw:  `{"memory":[{"id":"0","text":""},{"id":"1","text":"  "},{"id":"2","text":"keeper"}]}`,
			want: []string{"keeper"},
		},
		{
			name: "trims whitespace",
			raw:  `{"memory":[{"id":"0","text":"  spaced fact  "}]}`,
			want: []string{"spaced fact"},
		},
		{
			name: "garbage returns nil",
			raw:  `I could not produce JSON, sorry.`,
			want: nil,
		},
		{
			name: "truncated json returns nil",
			raw:  `{"memory":[{"id":"0","text":"unterminated`,
			want: nil,
		},
		{
			name: "empty string",
			raw:  ``,
			want: nil,
		},
		{
			name: "braces inside string value",
			raw:  `{"memory":[{"id":"0","text":"uses {curly} braces"}]}`,
			want: []string{"uses {curly} braces"},
		},
		{
			name: "multiple facts order preserved",
			raw:  `{"memory":[{"id":"0","text":"first"},{"id":"1","text":"second"},{"id":"2","text":"third"}]}`,
			want: []string{"first", "second", "third"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := parseExtraction(tt.raw)
			if len(got) != len(tt.want) {
				t.Fatalf("got %d facts %v, want %d %v", len(got), texts(got), len(tt.want), tt.want)
			}
			for i := range tt.want {
				if got[i].Text != tt.want[i] {
					t.Errorf("fact[%d] = %q, want %q", i, got[i].Text, tt.want[i])
				}
			}
		})
	}
}

func texts(fs []extractedFact) []string {
	out := make([]string, len(fs))
	for i, f := range fs {
		out[i] = f.Text
	}
	return out
}
