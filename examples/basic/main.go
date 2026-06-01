// Command basic is a runnable end-to-end dogfood of the mneme library against a
// real OpenAI-compatible endpoint. It constructs a Memory purely from MNEME_*
// environment variables, ingests a short conversation, then searches, gets and
// deletes — exercising the whole public surface a consumer touches.
//
//	export MNEME_LLM_BASE_URL=https://openrouter.ai/api/v1
//	export MNEME_LLM_API_KEY=$OPENROUTER_API_KEY
//	export MNEME_LLM_MODEL=openai/gpt-4o-mini
//	export MNEME_EMBED_MODEL=text-embedding-3-small
//	export MNEME_DB_PATH=/tmp/mneme-demo.db
//	go run ./examples/basic
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/AccursedGalaxy/mneme"
)

func main() {
	m, err := mneme.New() // providers + store built from MNEME_* env
	if err != nil {
		fmt.Fprintln(os.Stderr, "new:", err)
		os.Exit(1)
	}
	defer m.Close()

	ctx := context.Background()
	scope := mneme.Scope{UserID: "demo-user"}

	conv := []mneme.Message{
		{Role: "user", Content: "Hey! I just moved to Berlin and started a job as a backend engineer at Zalando. I also adopted a tabby cat named Pixel."},
		{Role: "assistant", Content: "Congrats on the move and the new role!"},
		{Role: "user", Content: "Thanks. Oh, and I'm allergic to shellfish, just so you know."},
	}

	written, err := m.Add(ctx, conv, scope)
	if err != nil {
		fmt.Fprintln(os.Stderr, "add:", err)
		os.Exit(1)
	}
	fmt.Printf("Extracted %d fact(s):\n", len(written))
	for _, f := range written {
		fmt.Printf("  • %s\n", f.Text)
	}

	for _, q := range []string{
		"where does the user live?",
		"what pets does the user have?",
		"any allergies I should know about?",
	} {
		hits, err := m.Search(ctx, q, scope, 1)
		if err != nil {
			fmt.Fprintln(os.Stderr, "search:", err)
			os.Exit(1)
		}
		fmt.Printf("\nQ: %s\n", q)
		for _, h := range hits {
			fmt.Printf("  → (%.3f) %s\n", h.Score, h.Text)
		}
	}

	// Round-trip Get and Delete on the first fact.
	if len(written) > 0 {
		id := written[0].ID
		got, err := m.Get(ctx, id)
		if err != nil {
			fmt.Fprintln(os.Stderr, "get:", err)
			os.Exit(1)
		}
		fmt.Printf("\nGet(%s) → %q\n", id, got.Text)
		if err := m.Delete(ctx, id); err != nil {
			fmt.Fprintln(os.Stderr, "delete:", err)
			os.Exit(1)
		}
		if _, err := m.Get(ctx, id); err != nil {
			fmt.Printf("Delete confirmed: fact %s is gone\n", id)
		}
	}
}
