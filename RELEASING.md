# Releasing

mneme has no version file — the version *is* the git tag (`go get` resolves it).
A release is therefore: land the changelog, tag, push. Publishing is automated.

## Cut a release

1. **Move `[Unreleased]` into a version section** in `CHANGELOG.md`: rename the
   heading to `## [X.Y.Z] - YYYY-MM-DD`, leave a fresh empty `[Unreleased]`
   above it, and update the compare links at the bottom of the file.
2. **Pick the bump** (SemVer, against the latest tag):
   - **patch** — fixes only;
   - **minor** — backward-compatible features (the usual case while `0.x`);
   - **major** — breaking public-API changes.
3. **Commit** the changelog (`release: vX.Y.Z`), then **tag and push**:
   ```sh
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin master vX.Y.Z
   ```

The `/ship` slash command automates steps 1–3 (verify green → changelog → tag →
push), so in practice this is just `/ship minor`.

## What happens on tag push

`.github/workflows/release.yml` triggers on any `v*` tag and:

1. re-runs the full quality bar (gofmt, vet, build, `go test -race`) as a
   release gate — a red tag never ships;
2. extracts the matching `## [X.Y.Z]` section from `CHANGELOG.md`;
3. creates the GitHub release with those notes (tags like `vX.Y.Z-rc.1` are
   marked pre-release automatically).

No manual step in the GitHub UI is needed. Keep `CHANGELOG.md` current and the
release notes take care of themselves.
