package bayes

import (
	"errors"
	"testing"

	"github.com/hickeroar/gobayes/v3/bayes/category"
)

// TestTrainUntrainLifecycle verifies train untrain lifecycle.
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

// TestClassifyAndScore verifies classify and score.
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

// TestEmptyAndUnknownInput verifies empty and unknown input.
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

// TestClassifyTieBreaksDeterministically verifies classify tie breaks deterministically.
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

// TestTrainReturnsErrorForInvalidCategoryNames verifies train returns error for invalid category names.
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

// TestUntrainReturnsErrorForInvalidCategoryNames verifies untrain returns error for invalid category names.
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

// TestFlushClearsCategories verifies flush clears categories.
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

// TestCustomTokenizerIsUsed verifies custom tokenizer is used.
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

// TestSummariesReturnsSnapshot verifies summaries returns snapshot.
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

// TestCalculateBayesianProbabilityDenominatorZero verifies calculate bayesian probability denominator zero.
func TestCalculateBayesianProbabilityDenominatorZero(t *testing.T) {
	classifier := NewClassifier()
	cat := category.NewCategory("x")
	got := classifier.calculateBayesianProbability(*cat, 0, 1)
	if got != 0.0 {
		t.Fatalf("expected 0 probability when denominator is zero, got %f", got)
	}
}

// TestNewClassifierWithTokenizer verifies custom tokenizer constructor.
func TestNewClassifierWithTokenizer(t *testing.T) {
	tok := func(s string) []string { return []string{"x", "y"} }
	c := NewClassifierWithTokenizer(tok)
	if c.Tokenizer == nil {
		t.Fatal("expected non-nil tokenizer")
	}
	if err := c.Train("a", "ignore"); err != nil {
		t.Fatalf("train failed: %v", err)
	}
	cat, _ := c.categories.LookupCategory("a")
	if cat.GetTokenCount("x") != 1 || cat.GetTokenCount("y") != 1 {
		t.Fatalf("expected custom tokenizer output, got counts x=%d y=%d", cat.GetTokenCount("x"), cat.GetTokenCount("y"))
	}
}

// TestNewClassifierWithOptions verifies options constructor and tokenizer config.
func TestNewClassifierWithOptions(t *testing.T) {
	c := NewClassifierWithOptions("spanish", true)
	if c.tokenizerLang != "spanish" {
		t.Fatalf("expected tokenizerLang spanish, got %q", c.tokenizerLang)
	}
	if !c.tokenizerRemoveStopWords {
		t.Fatal("expected tokenizerRemoveStopWords true")
	}
	if err := c.Train("spam", "el gato"); err != nil {
		t.Fatalf("train failed: %v", err)
	}
	cat, _ := c.categories.LookupCategory("spam")
	if cat.GetTokenCount("gat") == 0 && cat.GetTokenCount("gato") == 0 {
		t.Fatalf("expected spanish stemming to produce token")
	}
}

// TestNewClassifierWithOptionsNormalizesLang verifies language normalization.
func TestNewClassifierWithOptionsNormalizesLang(t *testing.T) {
	c := NewClassifierWithOptions("  SPANISH  ", false)
	if c.tokenizerLang != "spanish" {
		t.Fatalf("expected normalized lang spanish, got %q", c.tokenizerLang)
	}
	c2 := NewClassifierWithOptions("", false)
	if c2.tokenizerLang != "english" {
		t.Fatalf("expected default lang english for empty, got %q", c2.tokenizerLang)
	}
}

// TestDefaultTokenizerWithStopWords verifies stop-word filtering when enabled.
func TestDefaultTokenizerWithStopWords(t *testing.T) {
	tok := NewDefaultTokenizer("english", true)
	tokens := tok("the quick brown fox")
	for _, tkn := range tokens {
		if tkn == "the" {
			t.Fatal("expected 'the' to be filtered as stop word")
		}
	}
}

// TestDefaultTokenizerEmptyLangDefaultsToEnglish verifies empty or whitespace-only language defaults to english.
func TestDefaultTokenizerEmptyLangDefaultsToEnglish(t *testing.T) {
	for _, input := range []string{"", "   ", "\t"} {
		tok := NewDefaultTokenizer(input, false)
		tokens := tok("test word")
		if len(tokens) < 2 {
			t.Fatalf("expected tokens for lang %q, got %v", input, tokens)
		}
	}
}

// TestDefaultTokenizerUnsupportedLangDefaultsToEnglish verifies unsupported language falls back to english.
func TestDefaultTokenizerUnsupportedLangDefaultsToEnglish(t *testing.T) {
	for _, lang := range []string{"klingon", "xyz", "nosuchlang"} {
		tok := NewDefaultTokenizer(lang, false)
		tokens := tok("Running runs")
		if len(tokens) < 2 {
			t.Fatalf("expected stemming with english fallback for %q, got %v", lang, tokens)
		}
	}
}

// TestDefaultTokenizerAllLanguages exercises all seven supported languages.
func TestDefaultTokenizerAllLanguages(t *testing.T) {
	langs := []string{"english", "spanish", "french", "russian", "swedish", "norwegian", "hungarian"}
	for _, lang := range langs {
		tok := NewDefaultTokenizer(lang, false)
		tokens := tok("test word")
		if len(tokens) == 0 {
			t.Fatalf("expected tokens for %s, got none", lang)
		}
	}
}

// TestDefaultTokenizerNormalizesPunctuationAndStems verifies default tokenizer normalizes punctuation and stems.
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
