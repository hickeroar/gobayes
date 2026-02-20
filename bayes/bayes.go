package bayes

import (
	"sort"
	"strings"

	"github.com/hickeroar/gobayes/bayes/category"
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

// countTokenOccurances counts token frequencies in a token slice.
func (c *Classifier) countTokenOccurances(tokens []string) map[string]int {
	occurances := make(map[string]int)

	for _, token := range tokens {
		occurances[token]++
	}

	return occurances
}

// calculateCategoryProbabilities updates each category prior probability values.
func (c *Classifier) calculateCategoryProbabilities() {
	totalTally := 0.0
	probabilities := make(map[string]float64)

	// Tallying up the tallies for each category
	for _, name := range c.Categories.Names() {
		cat, ok := c.Categories.LookupCategory(name)
		if !ok {
			continue
		}
		probabilities[name] = float64(cat.GetTally())
		totalTally += probabilities[name]
	}

	// Calculating the probability that any given token is in each category
	for name, count := range probabilities {
		if totalTally > 0.0 {
			probabilities[name] = count / totalTally
		} else {
			probabilities[name] = 0.0
		}
	}

	// Calculating the probability that any given token is NOT in each category, and storing values on the category
	for name, probability := range probabilities {
		c.Categories.SetCategoryProbabilities(name, probability, 1.0-probability)
	}
}

// Flush resets all trained categories.
func (c *Classifier) Flush() {
	c.Categories = *category.NewCategories()
}

// Train updates a category with token counts from a text sample.
func (c *Classifier) Train(category string, text string) {
	cat := c.Categories.GetCategory(category)

	tokens := c.getTokenizer()(text)
	occurances := c.countTokenOccurances(tokens)

	for token, count := range occurances {
		if err := cat.TrainToken(token, count); err != nil {
			continue
		}
	}

	c.cleanUpCategory(cat)
	c.calculateCategoryProbabilities()
}

// Untrain removes token counts from a category using a text sample.
func (c *Classifier) Untrain(category string, text string) {
	cat := c.Categories.GetCategory(category)

	tokens := c.getTokenizer()(text)
	occurances := c.countTokenOccurances(tokens)

	for token, count := range occurances {
		if err := cat.UntrainToken(token, count); err != nil {
			continue
		}
	}

	c.cleanUpCategory(cat)
	c.calculateCategoryProbabilities()
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
	occurances := c.countTokenOccurances(tokens)

	// Map to hold all scores for all categories
	scores := make(map[string]float64)
	categoryNames := c.Categories.Names()
	for _, key := range categoryNames {
		scores[key] = 0.0
	}

	// Looping through each string token and calculating its bayesian probability
	for word, count := range occurances {
		tokenScores := make(map[string]float64)
		tokenTally := 0.0

		// Getting the tallies of this token from all categories
		for _, name := range categoryNames {
			cat, ok := c.Categories.LookupCategory(name)
			if !ok {
				continue
			}
			tokenScores[name] = float64(cat.GetTokenCount(word))
			tokenTally += tokenScores[name]
		}

		// If this word had no occurances in any of our categories, we continue
		if tokenTally == 0.0 {
			continue
		}

		for name, tokenScore := range tokenScores {
			cat, ok := c.Categories.LookupCategory(name)
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
