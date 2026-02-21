# Contributing

## Local development checks

Run these before opening a PR:

```sh
go test ./...
go test -race ./...
go vet ./...
staticcheck ./...
govulncheck ./...
```

Optional but recommended:

```sh
go test -run=^$ -fuzz=FuzzCategoryFromPath -fuzztime=10s .
go test -run=^$ -fuzz=FuzzClassifyHandlerBody -fuzztime=10s .
go test -run=^$ -fuzz=FuzzClassifierInvariants -fuzztime=10s ./bayes
go test -run=^$ -bench='Benchmark(Train|Score|Classify)$' -benchmem ./bayes
go test -tags=integration -run '^TestIntegration' .
```

## CI parity

CI runs:
- `go test ./...`
- `go test -coverprofile=coverage.out ./...` with minimum coverage threshold
- `go test -race ./...`
- `go test -tags=integration -run '^TestIntegration' .`
- `go vet ./...`
- `staticcheck ./...`
- `govulncheck ./...`

CI also runs scheduled fuzz smoke tests.

## Release and versioning

- Use semantic version tags (for example, `v1.4.0`).
- Keep backward compatibility for existing HTTP routes and response JSON fields unless intentionally releasing a breaking change.
- If making a breaking module API change in the future, migrate to a new major module path (for example, `/v2`) before tagging the release.
