package bayes

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/hickeroar/gobayes/v3/bayes/category"
)

const persistedModelVersion = 1
const defaultModelFilePath = "/tmp/gobayes-model.json"

type tempFile interface {
	io.Writer
	Sync() error
	Close() error
	Name() string
}

var (
	errNilWriter            = errors.New("writer is nil")
	errNilReader            = errors.New("reader is nil")
	errPathNotAbsolute      = errors.New("path must be absolute")
	errUnsupportedVersion   = errors.New("unsupported model version")
	errInvalidCategoryName  = errors.New("invalid category name in persisted model")
	errInvalidTokenCount    = errors.New("invalid token count in persisted model")
	errInvalidCategoryTally = errors.New("invalid category tally in persisted model")
	createTemp              = func(dir, pattern string) (tempFile, error) { return os.CreateTemp(dir, pattern) }
	renameFile              = os.Rename
	removeFile              = os.Remove
)

type persistedTokenizer struct {
	Language        string `json:"language"`
	RemoveStopWords bool   `json:"removeStopWords"`
}

type modelState struct {
	Version    int                                    `json:"version"`
	Categories map[string]category.PersistedCategory `json:"categories"`
	Tokenizer  *persistedTokenizer                   `json:"tokenizer,omitempty"`
}

// Save writes classifier model data to a writer using JSON encoding.
func (c *Classifier) Save(w io.Writer) error {
	if w == nil {
		return errNilWriter
	}

	c.mu.RLock()
	state := modelState{
		Version:    persistedModelVersion,
		Categories: c.categories.ExportStates(),
	}
	if c.tokenizerLang != "" {
		state.Tokenizer = &persistedTokenizer{
			Language:        c.tokenizerLang,
			RemoveStopWords: c.tokenizerRemoveStopWords,
		}
	}
	c.mu.RUnlock()

	if err := json.NewEncoder(w).Encode(state); err != nil {
		return fmt.Errorf("encode model: %w", err)
	}

	return nil
}

// Load reads classifier model data from a JSON reader and replaces state.
func (c *Classifier) Load(r io.Reader) error {
	if r == nil {
		return errNilReader
	}

	var state modelState
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&state); err != nil {
		return fmt.Errorf("decode model: %w", err)
	}

	if err := validateModelState(state); err != nil {
		return err
	}

	cats := category.NewCategories()
	_ = cats.ReplaceStates(state.Categories)
	cats.EnsureCategoryProbabilities()

	c.mu.Lock()
	c.categories = *cats
	if state.Tokenizer != nil {
		lang := strings.ToLower(strings.TrimSpace(state.Tokenizer.Language))
		if lang == "" {
			lang = "english"
		}
		c.Tokenizer = NewDefaultTokenizer(lang, state.Tokenizer.RemoveStopWords)
		c.tokenizerLang = lang
		c.tokenizerRemoveStopWords = state.Tokenizer.RemoveStopWords
	}
	c.mu.Unlock()

	return nil
}

// SaveToFile writes classifier model data to a file atomically.
func (c *Classifier) SaveToFile(path string) error {
	path = resolveModelPath(path)
	if !filepath.IsAbs(path) {
		return fmt.Errorf("%w: %q", errPathNotAbsolute, path)
	}

	dir := filepath.Dir(path)
	tempFile, err := createTemp(dir, ".gobayes-*")
	if err != nil {
		return fmt.Errorf("create temp file: %w", err)
	}
	tempPath := tempFile.Name()
	defer removeFile(tempPath)

	if err := c.Save(tempFile); err != nil {
		tempFile.Close()
		return err
	}
	if err := tempFile.Sync(); err != nil {
		tempFile.Close()
		return fmt.Errorf("sync temp file: %w", err)
	}
	if err := tempFile.Close(); err != nil {
		return fmt.Errorf("close temp file: %w", err)
	}

	if err := renameFile(tempPath, path); err != nil {
		return fmt.Errorf("rename temp file: %w", err)
	}

	return nil
}

// LoadFromFile reads classifier model data from a JSON file.
func (c *Classifier) LoadFromFile(path string) error {
	path = resolveModelPath(path)
	if !filepath.IsAbs(path) {
		return fmt.Errorf("%w: %q", errPathNotAbsolute, path)
	}

	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open model file: %w", err)
	}
	defer f.Close()

	return c.Load(f)
}

// validateModelState validates persisted classifier state before loading.
func validateModelState(state modelState) error {
	if state.Version != persistedModelVersion {
		return fmt.Errorf("%w: %d", errUnsupportedVersion, state.Version)
	}

	for name, cat := range state.Categories {
		if !categoryNamePattern.MatchString(name) {
			return fmt.Errorf("%w: %q", errInvalidCategoryName, name)
		}

		if cat.Tally < 0 {
			return fmt.Errorf("%w for %q: %d", errInvalidCategoryTally, name, cat.Tally)
		}

		sum := 0
		for token, count := range cat.Tokens {
			if token == "" || count <= 0 {
				return fmt.Errorf("%w for %q token %q: %d", errInvalidTokenCount, name, token, count)
			}
			sum += count
		}

		if sum != cat.Tally {
			return fmt.Errorf("%w for %q: tally=%d sum=%d", errInvalidCategoryTally, name, cat.Tally, sum)
		}
	}

	return nil
}

// resolveModelPath returns the default model path when no path is provided.
func resolveModelPath(path string) string {
	if path == "" {
		return defaultModelFilePath
	}
	return path
}
