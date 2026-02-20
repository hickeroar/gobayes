package bayes

import (
	"strings"
	"testing"
)

func TestTrainUntrainLifecycle(t *testing.T) {
	classifier := NewClassifier()

	classifier.Train("spam", "buy now buy now")
	spam, ok := classifier.Categories.LookupCategory("spam")
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
	spam, ok = classifier.Categories.LookupCategory("spam")
	if !ok {
		t.Fatal("expected spam category to still exist")
	}
	if spam.GetTally() != 2 {
		t.Fatalf("unexpected spam tally after untrain: got %d, want 2", spam.GetTally())
	}

	classifier.Untrain("spam", "buy now")
	if _, ok := classifier.Categories.LookupCategory("spam"); ok {
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

func FuzzClassifierInvariants(f *testing.F) {
	f.Add("spam", "buy now buy now")
	f.Add("ham", "hello world")
	f.Add("tech", "")

	f.Fuzz(func(t *testing.T, category string, sample string) {
		classifier := NewClassifier()

		if category == "" {
			category = "default"
		}
		classifier.Train(category, sample)
		classifier.Untrain(category, sample)
		classifier.Train(category, sample+" "+sample)
		_ = classifier.Score(sample)
		_ = classifier.Classify(sample)

		for _, name := range classifier.Categories.Names() {
			cat, ok := classifier.Categories.LookupCategory(name)
			if !ok {
				continue
			}
			if cat.GetTally() < 0 {
				t.Fatalf("category %q has negative tally: %d", name, cat.GetTally())
			}
			if cat.GetProbInCat() < 0 || cat.GetProbInCat() > 1 {
				t.Fatalf("category %q has invalid probIn: %f", name, cat.GetProbInCat())
			}
			if cat.GetProbNotInCat() < 0 || cat.GetProbNotInCat() > 1 {
				t.Fatalf("category %q has invalid probNotIn: %f", name, cat.GetProbNotInCat())
			}
			sum := cat.GetProbInCat() + cat.GetProbNotInCat()
			if sum < 0.999999 || sum > 1.000001 {
				t.Fatalf("category %q has invalid probability sum: %f", name, sum)
			}
		}
	})
}

func buildBenchmarkClassifier() *Classifier {
	classifier := NewClassifier()
	classifier.Train("tech", strings.Repeat("kubernetes latency tracing retries ", 50))
	classifier.Train("finance", strings.Repeat("portfolio rebalancing volatility alpha beta ", 50))
	classifier.Train("cooking", strings.Repeat("simmer saute reduction stock umami ", 50))
	return classifier
}

func BenchmarkTrain(b *testing.B) {
	classifier := NewClassifier()
	sample := strings.Repeat("distributed systems retries idempotency ", 20)

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		classifier.Train("tech", sample)
	}
}

func BenchmarkScore(b *testing.B) {
	classifier := buildBenchmarkClassifier()
	sample := "portfolio volatility and latency retries under stress"

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = classifier.Score(sample)
	}
}

func BenchmarkClassify(b *testing.B) {
	classifier := buildBenchmarkClassifier()
	sample := "simmer stock reduction with balanced acidity"

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = classifier.Classify(sample)
	}
}
