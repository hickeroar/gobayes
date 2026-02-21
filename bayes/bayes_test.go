package bayes

import (
	"errors"
	"testing"

	"github.com/hickeroar/gobayes/v3/bayes/category"
)

func TestTrainUntrainLifecycle(t *testing.T) {
	classifier := NewClassifier()

	if err := classifier.Train("spam", "buy now buy now"); err != nil {
		t.Fatalf("unexpected train error: %v", err)
	}
	spam, ok := classifier.categories.LookupCategory("spam")
	if !ok {
		t.Fatal("expected spam category to exist after training")
	}
	if spam.GetTally() != 4 {
		t.Fatalf("unexpected spam tally: got %d, want 4", spam.GetTally())
	}
	if spam.GetTokenCount("buy") != 2 {
		t.Fatalf("unexpected buy token count: got %d, want 2", spam.GetTokenCount("buy"))
	}

	if err := classifier.Untrain("spam", "buy now"); err != nil {
		t.Fatalf("unexpected untrain error: %v", err)
	}
	spam, ok = classifier.categories.LookupCategory("spam")
	if !ok {
		t.Fatal("expected spam category to still exist")
	}
	if spam.GetTally() != 2 {
		t.Fatalf("unexpected spam tally after untrain: got %d, want 2", spam.GetTally())
	}

	if err := classifier.Untrain("spam", "buy now"); err != nil {
		t.Fatalf("unexpected untrain error: %v", err)
	}
	if _, ok := classifier.categories.LookupCategory("spam"); ok {
		t.Fatal("expected spam category to be removed when tally reaches zero")
	}
}

func TestClassifyAndScore(t *testing.T) {
	classifier := NewClassifier()
	if err := classifier.Train("spam", "free prize click now"); err != nil {
		t.Fatalf("unexpected train error: %v", err)
	}
	if err := classifier.Train("ham", "team meeting schedule project"); err != nil {
		t.Fatalf("unexpected train error: %v", err)
	}

	classification := classifier.Classify("free prize now")
	if classification.Category != "spam" {
		t.Fatalf("unexpected classification category: got %q, want %q", classification.Category, "spam")
	}
	if classification.Score <= 0 {
		t.Fatalf("expected positive classification score, got %f", classification.Score)
	}

	scores := classifier.Score("meeting schedule")
	if len(scores) == 0 {
		t.Fatal("expected non-empty score map")
	}
	if scores["ham"] <= scores["spam"] {
		t.Fatalf("expected ham score to be greater for ham text: ham=%f spam=%f", scores["ham"], scores["spam"])
	}
}

func TestEmptyAndUnknownInput(t *testing.T) {
	classifier := NewClassifier()

	classification := classifier.Classify("")
	if classification.Category != "" {
		t.Fatalf("expected empty category for empty classifier, got %q", classification.Category)
	}

	if err := classifier.Train("ham", "hello world"); err != nil {
		t.Fatalf("unexpected train error: %v", err)
	}
	scores := classifier.Score("unseen tokens only")
	if len(scores) != 0 {
		t.Fatalf("expected no scores for unknown tokens, got %d entries", len(scores))
	}
}

func TestClassifyTieBreaksDeterministically(t *testing.T) {
	classifier := NewClassifier()
	if err := classifier.Train("alpha", "shared"); err != nil {
		t.Fatalf("unexpected train error: %v", err)
	}
	if err := classifier.Train("zeta", "shared"); err != nil {
		t.Fatalf("unexpected train error: %v", err)
	}

	classification := classifier.Classify("shared")
	if classification.Category != "alpha" {
		t.Fatalf("expected deterministic lexical tie break to alpha, got %q", classification.Category)
	}
}

func TestTrainReturnsErrorForInvalidCategoryNames(t *testing.T) {
	classifier := NewClassifier()
	if err := classifier.Train("", "hello world"); !errors.Is(err, ErrInvalidCategoryName) {
		t.Fatalf("expected ErrInvalidCategoryName for empty category, got %v", err)
	}
	if err := classifier.Train("spam!", "hello world"); !errors.Is(err, ErrInvalidCategoryName) {
		t.Fatalf("expected ErrInvalidCategoryName for invalid category, got %v", err)
	}

	if len(classifier.categories.Names()) != 0 {
		t.Fatalf("expected no categories for invalid names, got %v", classifier.categories.Names())
	}
}

func TestUntrainReturnsErrorForInvalidCategoryNames(t *testing.T) {
	classifier := NewClassifier()
	if err := classifier.Train("spam", "buy now"); err != nil {
		t.Fatalf("unexpected train error: %v", err)
	}
	if err := classifier.Untrain("", "buy now"); !errors.Is(err, ErrInvalidCategoryName) {
		t.Fatalf("expected ErrInvalidCategoryName for empty category, got %v", err)
	}
	if err := classifier.Untrain("spam!", "buy now"); !errors.Is(err, ErrInvalidCategoryName) {
		t.Fatalf("expected ErrInvalidCategoryName for invalid category, got %v", err)
	}

	cat, ok := classifier.categories.LookupCategory("spam")
	if !ok {
		t.Fatal("expected spam category to remain")
	}
	if got := cat.GetTally(); got != 2 {
		t.Fatalf("expected spam tally unchanged, got %d", got)
	}
}

func TestFlushClearsCategories(t *testing.T) {
	classifier := NewClassifier()
	if err := classifier.Train("spam", "buy now"); err != nil {
		t.Fatalf("unexpected train error: %v", err)
	}
	classifier.Flush()
	if got := len(classifier.categories.Names()); got != 0 {
		t.Fatalf("expected zero categories after flush, got %d", got)
	}
}

func TestCustomTokenizerIsUsed(t *testing.T) {
	classifier := NewClassifier()
	classifier.Tokenizer = func(string) []string {
		return []string{"custom", "custom", "token"}
	}
	if err := classifier.Train("tech", "ignored"); err != nil {
		t.Fatalf("unexpected train error: %v", err)
	}

	cat, ok := classifier.categories.LookupCategory("tech")
	if !ok {
		t.Fatal("expected tech category")
	}
	if got := cat.GetTokenCount("custom"); got != 2 {
		t.Fatalf("expected custom token count 2, got %d", got)
	}
}

func TestSummariesReturnsSnapshot(t *testing.T) {
	classifier := NewClassifier()
	if err := classifier.Train("spam", "buy now"); err != nil {
		t.Fatalf("unexpected train error: %v", err)
	}
	if err := classifier.Train("ham", "team meeting project"); err != nil {
		t.Fatalf("unexpected train error: %v", err)
	}

	summaries := classifier.Summaries()
	if len(summaries) != 2 {
		t.Fatalf("expected 2 category summaries, got %d", len(summaries))
	}
	if summaries["spam"].TokenTally != 2 {
		t.Fatalf("expected spam tally 2, got %d", summaries["spam"].TokenTally)
	}
	if summaries["ham"].TokenTally != 3 {
		t.Fatalf("expected ham tally 3, got %d", summaries["ham"].TokenTally)
	}
}

func TestCalculateBayesianProbabilityDenominatorZero(t *testing.T) {
	classifier := NewClassifier()
	cat := category.NewCategory("x")
	got := classifier.calculateBayesianProbability(*cat, 0, 1)
	if got != 0.0 {
		t.Fatalf("expected 0 probability when denominator is zero, got %f", got)
	}
}

func TestDefaultTokenizerNormalizesPunctuationAndStems(t *testing.T) {
	classifier := NewClassifier()
	if err := classifier.Train("run", "Running, RUNS! runner?"); err != nil {
		t.Fatalf("unexpected train error: %v", err)
	}

	cat, ok := classifier.categories.LookupCategory("run")
	if !ok {
		t.Fatal("expected run category")
	}

	if got := cat.GetTokenCount("run"); got < 2 {
		t.Fatalf("expected stemming to accumulate run tokens, got %d", got)
	}
	if got := cat.GetTokenCount("running"); got != 0 {
		t.Fatalf("expected running token to be stemmed away, got %d", got)
	}
}
