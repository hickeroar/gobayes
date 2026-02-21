package bayes

import (
	"bytes"
	"encoding/gob"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/hickeroar/gobayes/v2/bayes/category"
)

func TestPersistenceRoundTrip(t *testing.T) {
	original := NewClassifier()
	original.Train("spam", "buy now limited offer click")
	original.Train("ham", "team meeting project update")

	query := "limited offer now"
	wantClass := original.Classify(query)
	wantScores := original.Score(query)

	var buf bytes.Buffer
	if err := original.Save(&buf); err != nil {
		t.Fatalf("save failed: %v", err)
	}

	loaded := NewClassifier()
	if err := loaded.Load(&buf); err != nil {
		t.Fatalf("load failed: %v", err)
	}

	gotClass := loaded.Classify(query)
	if gotClass != wantClass {
		t.Fatalf("classification mismatch after round-trip: got %+v want %+v", gotClass, wantClass)
	}

	gotScores := loaded.Score(query)
	if len(gotScores) != len(wantScores) {
		t.Fatalf("score map length mismatch after round-trip: got %d want %d", len(gotScores), len(wantScores))
	}
	for name, want := range wantScores {
		got, ok := gotScores[name]
		if !ok {
			t.Fatalf("missing score for %q after round-trip", name)
		}
		if got != want {
			t.Fatalf("score mismatch for %q after round-trip: got %f want %f", name, got, want)
		}
	}
}

func TestLoadReplacesExistingState(t *testing.T) {
	source := NewClassifier()
	source.Train("spam", "buy now")

	var buf bytes.Buffer
	if err := source.Save(&buf); err != nil {
		t.Fatalf("save failed: %v", err)
	}

	target := NewClassifier()
	target.Train("ham", "team meeting")
	if _, ok := target.categories.LookupCategory("ham"); !ok {
		t.Fatal("expected preexisting ham category")
	}

	if err := target.Load(&buf); err != nil {
		t.Fatalf("load failed: %v", err)
	}

	if _, ok := target.categories.LookupCategory("ham"); ok {
		t.Fatal("expected ham category to be removed by replace-all load")
	}
	if _, ok := target.categories.LookupCategory("spam"); !ok {
		t.Fatal("expected spam category after load")
	}
}

func TestLoadRejectsInvalidPersistedState(t *testing.T) {
	tests := []struct {
		name  string
		state modelState
	}{
		{
			name: "invalid category name",
			state: modelState{
				Version: persistedModelVersion,
				Categories: map[string]category.PersistedCategory{
					"spam!": {Tokens: map[string]int{"buy": 1}, Tally: 1},
				},
			},
		},
		{
			name: "negative token count",
			state: modelState{
				Version: persistedModelVersion,
				Categories: map[string]category.PersistedCategory{
					"spam": {Tokens: map[string]int{"buy": -1}, Tally: 0},
				},
			},
		},
		{
			name: "negative tally",
			state: modelState{
				Version: persistedModelVersion,
				Categories: map[string]category.PersistedCategory{
					"spam": {Tokens: map[string]int{"buy": 1}, Tally: -1},
				},
			},
		},
		{
			name: "empty token",
			state: modelState{
				Version: persistedModelVersion,
				Categories: map[string]category.PersistedCategory{
					"spam": {Tokens: map[string]int{"": 1}, Tally: 1},
				},
			},
		},
		{
			name: "tally mismatch",
			state: modelState{
				Version: persistedModelVersion,
				Categories: map[string]category.PersistedCategory{
					"spam": {Tokens: map[string]int{"buy": 2}, Tally: 1},
				},
			},
		},
		{
			name: "unsupported version",
			state: modelState{
				Version: persistedModelVersion + 1,
				Categories: map[string]category.PersistedCategory{
					"spam": {Tokens: map[string]int{"buy": 1}, Tally: 1},
				},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var buf bytes.Buffer
			if err := gob.NewEncoder(&buf).Encode(tc.state); err != nil {
				t.Fatalf("failed to encode test state: %v", err)
			}

			classifier := NewClassifier()
			if err := classifier.Load(&buf); err == nil {
				t.Fatal("expected load to fail for invalid persisted state")
			}
		})
	}
}

func TestEmptyModelRoundTrip(t *testing.T) {
	classifier := NewClassifier()
	var buf bytes.Buffer

	if err := classifier.Save(&buf); err != nil {
		t.Fatalf("save failed: %v", err)
	}

	loaded := NewClassifier()
	if err := loaded.Load(&buf); err != nil {
		t.Fatalf("load failed: %v", err)
	}

	if got := len(loaded.categories.Names()); got != 0 {
		t.Fatalf("expected no categories after empty round-trip, got %d", got)
	}
}

func TestSaveToFileAndLoadFromFile(t *testing.T) {
	classifier := NewClassifier()
	classifier.Train("tech", "latency retries tracing")

	modelPath := filepath.Join(t.TempDir(), "model.gob")
	if err := classifier.SaveToFile(modelPath); err != nil {
		t.Fatalf("save to file failed: %v", err)
	}
	if _, err := os.Stat(modelPath); err != nil {
		t.Fatalf("expected model file to exist: %v", err)
	}

	loaded := NewClassifier()
	if err := loaded.LoadFromFile(modelPath); err != nil {
		t.Fatalf("load from file failed: %v", err)
	}

	result := loaded.Classify("tracing latency")
	if result.Category != "tech" {
		t.Fatalf("expected loaded model to classify as tech, got %q", result.Category)
	}
}

func TestSaveLoadRejectRelativePaths(t *testing.T) {
	classifier := NewClassifier()
	classifier.Train("spam", "buy now")

	if err := classifier.SaveToFile("model.gob"); err == nil {
		t.Fatal("expected SaveToFile to reject relative path")
	}

	if err := classifier.LoadFromFile("model.gob"); err == nil {
		t.Fatal("expected LoadFromFile to reject relative path")
	}
}

func TestSaveLoadDefaultPath(t *testing.T) {
	classifier := NewClassifier()
	classifier.Train("spam", "buy now")

	defaultPath := "/tmp/gobayes.gob"
	_ = os.Remove(defaultPath)
	defer os.Remove(defaultPath)

	if err := classifier.SaveToFile(""); err != nil {
		t.Fatalf("expected SaveToFile to use default path, got error: %v", err)
	}

	loaded := NewClassifier()
	if err := loaded.LoadFromFile(""); err != nil {
		t.Fatalf("expected LoadFromFile to use default path, got error: %v", err)
	}

	result := loaded.Classify("buy now")
	if result.Category != "spam" {
		t.Fatalf("expected loaded default-path model to classify as spam, got %q", result.Category)
	}
}

func TestSaveAndLoadNilAndDecodeErrors(t *testing.T) {
	classifier := NewClassifier()
	if err := classifier.Save(nil); err == nil {
		t.Fatal("expected error for nil writer")
	}
	if err := classifier.Load(nil); err == nil {
		t.Fatal("expected error for nil reader")
	}
	if err := classifier.Load(strings.NewReader("not-gob")); err == nil {
		t.Fatal("expected decode error for invalid gob payload")
	}
}

type failWriter struct{}

func (failWriter) Write([]byte) (int, error) {
	return 0, errors.New("write failed")
}

func TestSaveWriterError(t *testing.T) {
	classifier := NewClassifier()
	classifier.Train("spam", "buy now")
	if err := classifier.Save(failWriter{}); err == nil {
		t.Fatal("expected save to fail for writer error")
	}
}

func TestSaveToFileErrors(t *testing.T) {
	classifier := NewClassifier()
	classifier.Train("spam", "buy now")

	if err := classifier.SaveToFile("relative.gob"); err == nil {
		t.Fatal("expected relative path error")
	}

	badPath := filepath.Join(t.TempDir(), "no-such-dir", "model.gob")
	if err := classifier.SaveToFile(badPath); err == nil {
		t.Fatal("expected create temp file error for missing dir")
	}
}

type failReadCloser struct{}

func (failReadCloser) Read([]byte) (int, error) {
	return 0, errors.New("read failed")
}

func (failReadCloser) Close() error {
	return nil
}

func TestLoadFromFileAndLoadReaderErrors(t *testing.T) {
	classifier := NewClassifier()
	if err := classifier.LoadFromFile("relative.gob"); err == nil {
		t.Fatal("expected relative path error")
	}
	if err := classifier.LoadFromFile("/tmp/does-not-exist-gobayes.gob"); err == nil {
		t.Fatal("expected open model file error")
	}
	if err := classifier.Load(io.Reader(failReadCloser{})); err == nil {
		t.Fatal("expected load decode/read error")
	}
}

type fakeTempFile struct {
	name     string
	writeErr error
	syncErr  error
	closeErr error
}

func (f *fakeTempFile) Write(p []byte) (int, error) {
	if f.writeErr != nil {
		return 0, f.writeErr
	}
	return len(p), nil
}

func (f *fakeTempFile) Sync() error {
	return f.syncErr
}

func (f *fakeTempFile) Close() error {
	return f.closeErr
}

func (f *fakeTempFile) Name() string {
	return f.name
}

func TestSaveToFileSyncCloseAndRenameErrors(t *testing.T) {
	classifier := NewClassifier()
	classifier.Train("spam", "buy now")

	origCreateTemp := createTemp
	origRenameFile := renameFile
	origRemoveFile := removeFile
	defer func() {
		createTemp = origCreateTemp
		renameFile = origRenameFile
		removeFile = origRemoveFile
	}()

	removeFile = func(string) error { return nil }
	renameFile = func(string, string) error { return nil }

	createTemp = func(string, string) (tempFile, error) {
		return &fakeTempFile{name: "/tmp/fake-write.gob", writeErr: errors.New("write failed")}, nil
	}
	if err := classifier.SaveToFile("/tmp/model.gob"); err == nil {
		t.Fatal("expected save/write error from SaveToFile")
	}

	createTemp = func(string, string) (tempFile, error) {
		return &fakeTempFile{name: "/tmp/fake-sync.gob", syncErr: errors.New("sync failed")}, nil
	}
	if err := classifier.SaveToFile("/tmp/model.gob"); err == nil {
		t.Fatal("expected sync error from SaveToFile")
	}

	createTemp = func(string, string) (tempFile, error) {
		return &fakeTempFile{name: "/tmp/fake-close.gob", closeErr: errors.New("close failed")}, nil
	}
	if err := classifier.SaveToFile("/tmp/model.gob"); err == nil {
		t.Fatal("expected close error from SaveToFile")
	}

	createTemp = func(string, string) (tempFile, error) {
		return &fakeTempFile{name: "/tmp/fake-rename.gob"}, nil
	}
	renameFile = func(string, string) error { return errors.New("rename failed") }
	if err := classifier.SaveToFile("/tmp/model.gob"); err == nil {
		t.Fatal("expected rename error from SaveToFile")
	}
}
