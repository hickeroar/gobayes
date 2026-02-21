package bayes

import (
	"testing"

	"github.com/hickeroar/gobayes/v2/bayes/category"
)

func TestTrainUntrainLifecycle(t *testing.T) {
	classifier := NewClassifier()

	classifier.Train("spam", "buy now buy now")
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

	classifier.Untrain("spam", "buy now")
	spam, ok = classifier.categories.LookupCategory("spam")
	if !ok {
		t.Fatal("expected spam category to still exist")
	}
	if spam.GetTally() != 2 {
		t.Fatalf("unexpected spam tally after untrain: got %d, want 2", spam.GetTally())
	}

	classifier.Untrain("spam", "buy now")
	if _, ok := classifier.categories.LookupCategory("spam"); ok {
		t.Fatal("expected spam category to be removed when tally reaches zero")
	}
}

func TestClassifyAndScore(t *testing.T) {
	classifier := NewClassifier()
	classifier.Train("spam", "free prize click now")
	classifier.Train("ham", "team meeting schedule project")

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

	classifier.Train("ham", "hello world")
	scores := classifier.Score("unseen tokens only")
	if len(scores) != 0 {
		t.Fatalf("expected no scores for unknown tokens, got %d entries", len(scores))
	}
}

func TestClassifyTieBreaksDeterministically(t *testing.T) {
	classifier := NewClassifier()
	classifier.Train("alpha", "shared")
	classifier.Train("zeta", "shared")

	classification := classifier.Classify("shared")
	if classification.Category != "alpha" {
		t.Fatalf("expected deterministic lexical tie break to alpha, got %q", classification.Category)
	}
}

func TestTrainIgnoresInvalidCategoryNames(t *testing.T) {
	classifier := NewClassifier()
	classifier.Train("", "hello world")
	classifier.Train("spam!", "hello world")

	if len(classifier.categories.Names()) != 0 {
		t.Fatalf("expected no categories for invalid names, got %v", classifier.categories.Names())
	}
}

func TestUntrainIgnoresInvalidCategoryNames(t *testing.T) {
	classifier := NewClassifier()
	classifier.Train("spam", "buy now")
	classifier.Untrain("", "buy now")
	classifier.Untrain("spam!", "buy now")

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
	classifier.Train("spam", "buy now")
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
	classifier.Train("tech", "ignored")

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
	classifier.Train("spam", "buy now")
	classifier.Train("ham", "team meeting project")

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
