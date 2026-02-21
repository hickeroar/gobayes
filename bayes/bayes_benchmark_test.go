package bayes

import (
	"strings"
	"testing"
)

func buildBenchmarkClassifier() *Classifier {
	classifier := NewClassifier()
	_ = classifier.Train("tech", strings.Repeat("kubernetes latency tracing retries ", 50))
	_ = classifier.Train("finance", strings.Repeat("portfolio rebalancing volatility alpha beta ", 50))
	_ = classifier.Train("cooking", strings.Repeat("simmer saute reduction stock umami ", 50))
	return classifier
}

func BenchmarkTrain(b *testing.B) {
	classifier := NewClassifier()
	sample := strings.Repeat("distributed systems retries idempotency ", 20)

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = classifier.Train("tech", sample)
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
