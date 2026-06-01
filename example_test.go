package mneme_test

import (
	"context"
	"fmt"

	"github.com/AccursedGalaxy/mneme"
	"github.com/AccursedGalaxy/mneme/provider/fake"
	"github.com/AccursedGalaxy/mneme/store/sqlite"
)

// Example shows the full library surface: construct a Memory, Add a
// conversation, then Search it back. It uses offline fakes so it runs as part
// of the test suite; a real consumer would call mneme.New() to build providers
// from MNEME_* env vars instead.
func Example() {
	st, _ := sqlite.Open(":memory:")

	m, err := mneme.New(
		mneme.WithStore(st),
		mneme.WithLLM(&fake.LLM{Responses: []string{
			fake.JSON("Alice drives a Ferrari 488 GTB"),
		}}),
		mneme.WithEmbedder(&fake.Embedder{}),
	)
	if err != nil {
		panic(err)
	}
	defer m.Close()

	ctx := context.Background()
	scope := mneme.Scope{UserID: "alice"}

	written, _ := m.Add(ctx, []mneme.Message{
		{Role: "user", Content: "I drive a Ferrari 488 GTB"},
	}, scope)
	fmt.Printf("wrote %d fact(s)\n", len(written))

	hits, _ := m.Search(ctx, "what Ferrari does Alice drive?", scope, 1)
	for _, h := range hits {
		fmt.Println(h.Text)
	}
	// Output:
	// wrote 1 fact(s)
	// Alice drives a Ferrari 488 GTB
}
