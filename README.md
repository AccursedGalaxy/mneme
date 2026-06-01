# mneme

A small, self-contained **agent memory** library for Go — drop it into any agent
to give it persistent, searchable long-term memory.

- **Library-first.** Import `github.com/AccursedGalaxy/mneme` directly. A thin
  HTTP + MCP server binary wrapping the same core lands later (see `PLAN.md`).
- **Self-contained.** No shared code with any consumer. Its own OpenAI-compatible
  LLM/embedding client, its own storage.
- **Single binary.** Default storage is pure-Go SQLite (no cgo). Pluggable for
  sqlite-vec / pgvector as you scale.
- **Our own prompts.** The extraction prompt is ours, evaluated against a
  built-in harness — not a vendored copy.

Status: **pre-v1.** The full build spec lives in [`PLAN.md`](./PLAN.md) — start there.

## License

MIT — see [`LICENSE`](./LICENSE).
