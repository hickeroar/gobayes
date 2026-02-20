# Changelog

All notable changes to this project are documented here.

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
- Initial Naive Bayes classifier implementation and category structures.
- Early HTTP API endpoints and request handlers for training and classification.
- Support for category scoring, untraining, and flush operations.

### Changed
- Iterative fixes and refinements across classifier behavior, imports, docs, formatting, and repository setup.
