# Local mirror of CI (.github/workflows). `make ci` is what the CI job runs;
# `make eval` is the live prompt-regression gate (needs an API key).

.PHONY: ci fmt fmt-check vet build test test-race eval

# What CI enforces on every push/PR — all offline.
ci: fmt-check vet build test-race

fmt:
	gofmt -w .

fmt-check:
	@unformatted="$$(gofmt -l .)"; \
	if [ -n "$$unformatted" ]; then \
		echo "Not gofmt-clean (run 'make fmt'):"; echo "$$unformatted"; exit 1; \
	fi

vet:
	go vet ./...

build:
	go build ./...

test:
	go test ./...

test-race:
	go test -race -count=1 ./...

# Live eval-regression gate: scores the extraction prompt against fixtures and
# fails if any version drops below the floor. Needs MNEME_LLM_API_KEY or
# OPENROUTER_API_KEY (see .env). The floor sits below the 0.97 baseline to absorb
# live-LLM noise (≈0.97 ± ~0.01) while still catching real regressions; override
# with MIN=.
MIN ?= 0.95
eval:
	go run ./cmd/eval -embedder openai -min-aggregate $(MIN)
