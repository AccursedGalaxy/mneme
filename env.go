package mneme

import (
	"fmt"
	"os"

	"github.com/AccursedGalaxy/mneme/provider"
	"github.com/AccursedGalaxy/mneme/provider/openai"
	"github.com/AccursedGalaxy/mneme/store"
	"github.com/AccursedGalaxy/mneme/store/sqlite"
)

// Environment variables read by New when a provider/store is not supplied via
// options. See PLAN.md §8.
const (
	envLLMBaseURL   = "MNEME_LLM_BASE_URL"
	envLLMAPIKey    = "MNEME_LLM_API_KEY"
	envLLMModel     = "MNEME_LLM_MODEL"
	envEmbedBaseURL = "MNEME_EMBED_BASE_URL"
	envEmbedAPIKey  = "MNEME_EMBED_API_KEY"
	envEmbedModel   = "MNEME_EMBED_MODEL"
	envDBPath       = "MNEME_DB_PATH"
)

// defaultDBPath is used when MNEME_DB_PATH is unset.
const defaultDBPath = "./mneme.db"

func llmFromEnv() (provider.LLM, error) {
	base := os.Getenv(envLLMBaseURL)
	model := os.Getenv(envLLMModel)
	if base == "" || model == "" {
		return nil, fmt.Errorf("mneme: no LLM provided and %s/%s not set "+
			"(pass mneme.WithLLM or set the env)", envLLMBaseURL, envLLMModel)
	}
	return &openai.LLM{
		BaseURL: base,
		APIKey:  os.Getenv(envLLMAPIKey),
		Model:   model,
	}, nil
}

func embedderFromEnv() (provider.Embedder, error) {
	base := os.Getenv(envEmbedBaseURL)
	if base == "" {
		base = os.Getenv(envLLMBaseURL) // embeddings default to the LLM base
	}
	model := os.Getenv(envEmbedModel)
	if base == "" || model == "" {
		return nil, fmt.Errorf("mneme: no embedder provided and %s/%s not set "+
			"(pass mneme.WithEmbedder or set the env)", envEmbedBaseURL, envEmbedModel)
	}
	key := os.Getenv(envEmbedAPIKey)
	if key == "" {
		key = os.Getenv(envLLMAPIKey)
	}
	return &openai.Embedder{
		BaseURL: base,
		APIKey:  key,
		Model:   model,
	}, nil
}

func storeFromEnv() (store.Store, error) {
	path := os.Getenv(envDBPath)
	if path == "" {
		path = defaultDBPath
	}
	return sqlite.Open(path)
}
