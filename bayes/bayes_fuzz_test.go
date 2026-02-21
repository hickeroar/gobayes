package bayes

import "testing"

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

		for _, name := range classifier.categories.Names() {
			cat, ok := classifier.categories.LookupCategory(name)
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
