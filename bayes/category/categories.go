package category

// CategorySummary is a read-only summary used by API responses.
type CategorySummary struct {
	TokenTally   int
	ProbNotInCat float64
	ProbInCat    float64
}

// Categories stores and manages trained Category values.
type Categories struct {
	categories         map[string]*Category // Map of category names to categories
	probabilitiesDirty bool
}

// NewCategories returns an initialized Categories collection.
func NewCategories() *Categories {
	return &Categories{
		categories:         make(map[string]*Category),
		probabilitiesDirty: true,
	}
}

// AddCategory creates and stores a new category by name.
func (cats *Categories) AddCategory(name string) *Category {
	cat := NewCategory(name)

	cats.categories[name] = cat
	cats.probabilitiesDirty = true

	return cat
}

// GetCategory returns the category by name, creating it when missing.
func (cats *Categories) GetCategory(name string) *Category {
	if val, ok := cats.categories[name]; ok {
		return val
	}

	// If we get here, we don't have this category, so we're adding it.
	return cats.AddCategory(name)
}

// DeleteCategory removes a category by name.
func (cats *Categories) DeleteCategory(name string) {
	delete(cats.categories, name)
	cats.probabilitiesDirty = true
}

// Names returns all known category names.
func (cats *Categories) Names() []string {
	names := make([]string, 0, len(cats.categories))
	for name := range cats.categories {
		names = append(names, name)
	}
	return names
}

// LookupCategory returns a category by name without creating one.
func (cats *Categories) LookupCategory(name string) (*Category, bool) {
	cat, ok := cats.categories[name]
	return cat, ok
}

// SetCategoryProbabilities updates derived probability fields for an existing category.
func (cats *Categories) SetCategoryProbabilities(name string, probInCat float64, probNotInCat float64) {
	if cat, ok := cats.categories[name]; ok {
		cat.setProbabilities(probInCat, probNotInCat)
	}
}

// MarkProbabilitiesDirty marks cached priors for recalculation.
func (cats *Categories) MarkProbabilitiesDirty() {
	cats.probabilitiesDirty = true
}

// EnsureCategoryProbabilities recalculates priors when data has changed.
func (cats *Categories) EnsureCategoryProbabilities() {
	if !cats.probabilitiesDirty {
		return
	}

	totalTally := 0.0
	probabilities := make(map[string]float64, len(cats.categories))

	for name, cat := range cats.categories {
		tally := float64(cat.GetTally())
		probabilities[name] = tally
		totalTally += tally
	}

	for name, tally := range probabilities {
		probability := 0.0
		if totalTally > 0.0 {
			probability = tally / totalTally
		}
		cats.SetCategoryProbabilities(name, probability, 1.0-probability)
	}

	cats.probabilitiesDirty = false
}

// Summaries returns a response-oriented snapshot of category data.
func (cats *Categories) Summaries() map[string]CategorySummary {
	cats.EnsureCategoryProbabilities()

	snapshot := make(map[string]CategorySummary, len(cats.categories))
	for name, cat := range cats.categories {
		snapshot[name] = CategorySummary{
			TokenTally:   cat.GetTally(),
			ProbNotInCat: cat.GetProbNotInCat(),
			ProbInCat:    cat.GetProbInCat(),
		}
	}
	return snapshot
}
