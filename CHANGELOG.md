# Changelog

All notable changes to this project are documented here.

## Unreleased

### Added
- CLI flags: `--host`, `--language`, `--remove-stop-words`, `--verbose` (existing `--port` and `--auth-token` unchanged).
- Environment variables for defaults: `GOBAYES_HOST`, `GOBAYES_PORT`, `GOBAYES_AUTH_TOKEN`, `GOBAYES_LANGUAGE`, `GOBAYES_REMOVE_STOP_WORDS`, `GOBAYES_VERBOSE`. Boolean env (REMOVE_STOP_WORDS, VERBOSE) accept `1`, `true`, or `yes` (case-insensitive) for enabled.
- Optional verbose logging: when `--verbose` or `GOBAYES_VERBOSE` is set, request/response and body previews are logged to stderr.
- Server bind address uses host and port (e.g. `0.0.0.0:8000` or `127.0.0.1:9000`); configurable via `--host` and `--port`.
- Classifier language and stop-word removal are configurable from CLI/env and applied via `NewClassifierWithOptions`.

### Changed
- Help/usage output shows double-dash form for all flags (e.g. `--port`). Single-dash forms (e.g. `-port`) remain accepted for backward compatibility.
- Default precedence: hard-coded default, then env var, then CLI flag (flag wins).

### Notes
- Backward compatible: existing scripts using `-port` or `-auth-token` continue to work. `/healthz` and `/readyz` remain unauthenticated when auth is enabled.

## v3.2.1

### Changed
- Documentation: package doc and changelog now use "Naive Bayesian" instead of "Naive Bayes" for terminology consistency.

## v3.2.0

### Added
- Multi-language tokenization for seven languages (english, spanish, french, russian, swedish, norwegian, hungarian) via kljensen/snowball.
- Optional stop-word filtering via self-contained `bayes/stopwords` package; opt-in, default off (aligned with simplebayes).
- `NewDefaultTokenizer(lang, removeStopWords)` factory for configurable tokenization.
- `NewClassifierWithOptions(lang, removeStopWords)` constructor; tokenizer config is persisted on Save and restored on Load.
- `NewClassifierWithTokenizer(tokenizer)` for custom tokenizers (config not persisted).
- Locale-aware lowercasing via `golang.org/x/text/cases` for correct case folding per language.
- Tokenizer metadata persisted in model JSON: `"tokenizer": {"language": "english", "removeStopWords": false}`.

### Changed
- JSON Load now uses `DisallowUnknownFields()` for stricter validation and resilience to malformed payloads.
- Default tokenizer uses locale-aware lowercasing instead of `strings.ToLower`.
- Persisted model format extended with optional `tokenizer` object; backward compatible with v3.1.0 saves (no tokenizer field).
- README updated for multi-language tokenization, stop words, and persisted tokenizer config.
- `coverage.html` added to `.gitignore`.

### Notes
- When snowball.Stem fails or returns empty, the original token is kept. Documented in package and tokenizer docs.
- Unsupported language names fall back to English stemming. Empty or whitespace-only language defaults to English.

## v3.1.0

### Changed
- Persistence format for library save/load APIs migrated from gob to JSON.
- Default persistence file path changed from `/tmp/gobayes.gob` to `/tmp/gobayes-model.json`.
- Persistence tests updated to validate JSON encode/decode behavior and JSON-based path fixtures.
- README persistence examples and notes updated to reference JSON persistence paths/files.

### Notes
- `Load` now expects JSON model payloads; legacy gob model files are no longer supported.
- Persisted model content remains category/token tallies with model version metadata; runtime probabilities are recomputed after load.

## v3.0.0

### Breaking
- Module path moved from `github.com/hickeroar/gobayes/v2` to `github.com/hickeroar/gobayes/v3`.
- Library API signatures changed:
  - `Classifier.Train(category, text string) error`
  - `Classifier.Untrain(category, text string) error`
- HTTP JSON response keys migrated from PascalCase to camelCase:
  - `Category` -> `category`
  - `Score` -> `score`
  - `Success` -> `success`
  - `Categories` -> `categories`
  - `TokenTally` -> `tokenTally`
  - `ProbInCat` -> `probInCat`
  - `ProbNotInCat` -> `probNotInCat`

### Added
- Default tokenizer pipeline now applies Unicode normalization and English stemming, while still allowing a custom tokenizer override via `Classifier.Tokenizer`.
- `bayes.ErrInvalidCategoryName` is returned when `Train`/`Untrain` receives a category that does not match `^[-_A-Za-z0-9]+$`.

### Migration Notes
- Update imports:
  - from `github.com/hickeroar/gobayes/v2/bayes`
  - to `github.com/hickeroar/gobayes/v3/bayes`
- Handle errors from `Train` and `Untrain` in library consumers.
- Update HTTP client decoders to camelCase keys.

Before:
```json
{"Category":"spam","Score":43.48}
```

After:
```json
{"category":"spam","score":43.48}
```

## v2.1.2

### Added
- `Classifier.Summaries()` as the public summary/snapshot accessor used by HTTP responses.
- Regression coverage for summary snapshots and idempotent probability refresh behavior.
- `coverage_cat.out` to `.gitignore` to keep local coverage artifacts out of commits.

### Changed
- Concurrency handling in classifier read paths: `Classify` and `Score` now use read locks.
- Probability refresh now occurs during state mutation/load paths instead of score-time recalculation.
- HTTP controller now keeps a pointer to `bayes.Classifier` and relies on classifier-level locking (removed redundant API-level mutex locking).
- Internal category state on `Classifier` is now encapsulated and no longer exposed as a public field.

### Notes
- HTTP endpoint behavior and JSON response shapes remain compatible.
- Go package consumers that previously accessed `Classifier.Categories` directly should migrate to public methods (including `Summaries()`).

## v2.1.1

### Added
- Subprocess-backed integration tests that start the real Gobayes server and validate end-to-end API behavior.
- Dedicated CI integration lane running `go test -tags=integration -run '^TestIntegration' .`.

### Changed
- Disabled `actions/setup-go` cache (`cache: false`) across CI jobs to avoid `go.sum` cache warnings for this stdlib-only module.
- CI workflow now triggers on `pull_request` (plus manual/scheduled runs), removing duplicate PR branch runs from `push` + `pull_request`.

## v2.1.0

### Added
- Optional webserver auth token via `--auth-token`.
- Bearer-token enforcement on API endpoints (`/info`, `/train/*`, `/untrain/*`, `/classify`, `/score`, `/flush`).

### Changed
- CLI examples/docs now prefer double-hyphen flag usage (`--port`, `--auth-token`) while keeping single-hyphen compatibility.
- Go baseline moved to `1.26` in `go.mod`.
- CI test matrix updated to Go `1.26.x` only.

### Notes
- `/healthz` and `/readyz` remain unauthenticated for probe compatibility.

## v2.0.0 (today's changes currently on trunk)

### Changed
- Major modernization of Gobayes internals and API behavior for safer defaults and maintainability.
- Added persistence and restore support for classifier state, with concurrency-safety improvements.
- Hardened quality gates and release semantics for the v2 module line.
- Pinned and validated Go CI/tooling lanes for current v2 development.
- Split tests into focused files by concern to improve maintainability and clarity.

## v1.0.0 (all changes before today)

### Added
- Initial Naive Bayesian classifier implementation and category structures.
- Early HTTP API endpoints and request handlers for training and classification.
- Support for category scoring, untraining, and flush operations.

### Changed
- Iterative fixes and refinements across classifier behavior, imports, docs, formatting, and repository setup.
