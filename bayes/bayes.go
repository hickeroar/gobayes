package bayes

import (
	"regexp"
	"sort"
	"strings"

	"github.com/hickeroar/gobayes/v2/bayes/category"
)

// Classification is the result of classifying a text sample.
type Classification struct {
	Category string  `json:"Category"`
	Score    float64 `json:"Score"`
}

// Classifier trains text categories and classifies new text samples.
//
// Classifier is not safe for concurrent use without external synchronization.
type Classifier struct {
	Categories category.Categories
	Tokenizer  func(string) []string
}

var categoryNamePattern = regexp.MustCompile(`^[-_A-Za-z0-9]+$`)

// NewClassifier returns a new Classifier instance.
func NewClassifier() *Classifier {
	return &Classifier{
		Categories: *category.NewCategories(),
	}
}

// tokenizeText lowercases and tokenizes text using whitespace separation.
func (c *Classifier) tokenizeText(sample string) []string {
	sample = strings.ToLower(sample)
	return strings.Fields(sample)
}

// getTokenizer returns the configured tokenizer or the default tokenizer.
func (c *Classifier) getTokenizer() func(string) []string {
	if c.Tokenizer == nil {
		return c.tokenizeText
	}
	return c.Tokenizer
}

// countTokenOccurrences counts token frequencies in a token slice.
func (c *Classifier) countTokenOccurrences(tokens []string) map[string]int {
	occurrences := make(map[string]int)

	for _, token := range tokens {
		occurrences[token]++
	}

	return occurrences
}

// Flush resets all trained categories.
func (c *Classifier) Flush() {
	c.Categories = *category.NewCategories()
}

// Train updates a category with token counts from a text sample.
func (c *Classifier) Train(category string, text string) {
	if !categoryNamePattern.MatchString(category) {
		return
	}

	cat := c.Categories.GetCategory(category)

	tokens := c.getTokenizer()(text)
	occurrences := c.countTokenOccurrences(tokens)

	for token, count := range occurrences {
		if err := cat.TrainToken(token, count); err != nil {
			continue
		}
	}

	c.cleanUpCategory(cat)
	c.Categories.MarkProbabilitiesDirty()
}

// Untrain removes token counts from a category using a text sample.
func (c *Classifier) Untrain(category string, text string) {
	if !categoryNamePattern.MatchString(category) {
		return
	}

	cat := c.Categories.GetCategory(category)

	tokens := c.getTokenizer()(text)
	occurrences := c.countTokenOccurrences(tokens)

	for token, count := range occurrences {
		if err := cat.UntrainToken(token, count); err != nil {
			continue
		}
	}

	c.cleanUpCategory(cat)
	c.Categories.MarkProbabilitiesDirty()
}

// cleanUpCategory removes an empty category.
func (c *Classifier) cleanUpCategory(cat *category.Category) {
	// If there are no tokens in this category, we delete the category.
	if cat.GetTally() == 0 {
		c.Categories.DeleteCategory(cat.Name())
	}
}

// Classify scores text against all categories and returns the best match.
func (c *Classifier) Classify(text string) Classification {
	scores := c.Score(text)
	result := Classification{}

	// If we had no scores returned we just return the Classification object without a category
	if len(scores) == 0 {
		return result
	}

	names := make([]string, 0, len(scores))
	for name := range scores {
		names = append(names, name)
	}
	sort.Strings(names)

	for _, name := range names {
		score := scores[name]
		if score > result.Score {
			result.Category = name
			result.Score = score
		}
	}

	return result
}

// Score computes Bayesian scores for each category given a text sample.
func (c *Classifier) Score(text string) map[string]float64 {

	tokens := c.getTokenizer()(text)
	occurrences := c.countTokenOccurrences(tokens)
	c.Categories.EnsureCategoryProbabilities()

	// Map to hold all scores for all categories
	scores := make(map[string]float64)
	categoryNames := c.Categories.Names()
	categoriesByName := make(map[string]*category.Category, len(categoryNames))
	for _, key := range categoryNames {
		cat, ok := c.Categories.LookupCategory(key)
		if !ok {
			continue
		}
		categoriesByName[key] = cat
		scores[key] = 0.0
	}

	// Looping through each string token and calculating its bayesian probability
	for word, count := range occurrences {
		tokenScores := make(map[string]float64)
		tokenTally := 0.0

		// Getting the tallies of this token from all categories
		for name, cat := range categoriesByName {
			tokenScores[name] = float64(cat.GetTokenCount(word))
			tokenTally += tokenScores[name]
		}

		// If this word had no occurrences in any of our categories, we continue
		if tokenTally == 0.0 {
			continue
		}

		for name, tokenScore := range tokenScores {
			cat, ok := categoriesByName[name]
			if !ok {
				continue
			}
			probability := c.calculateBayesianProbability(*cat, tokenScore, tokenTally)
			fcount := float64(count)
			scores[name] += fcount * probability
		}
	}

	// Only including scores that are greater than 0
	finalScores := make(map[string]float64)
	for name, score := range scores {
		if score > 0.0 {
			finalScores[name] = score
		}
	}
	return finalScores
}

// calculateBayesianProbability computes the Bayesian probability for one category.
func (c *Classifier) calculateBayesianProbability(category category.Category, tokenScore float64, tokenTally float64) float64 {
	// P that any given token IS in this category
	prc := category.GetProbInCat()
	// P that any given token is NOT in this category
	prnc := category.GetProbNotInCat()
	// P that this token is NOT of this category
	tokPrnc := (tokenTally - tokenScore) / tokenTally
	// P that this token IS of this category
	tokPrc := tokenScore / tokenTally

	numerator := tokPrc * prc
	denominator := (tokPrnc * prnc) + numerator

	if denominator != 0.0 {
		return numerator / denominator
	}

	// Default value if the denominator comes out to 0
	return 0.0
}
