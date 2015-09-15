package bayes

import (
	"github.com/hickeroar/gobayes/bayes/category"
	"strings"
)

// Classification is the result object from a classify action against the Classifier struct
type Classification struct {
	Category category.Category
	Score    float64
}

// Classifier is responsible for classifying text samples
type Classifier struct {
	Categories category.Categories
	Tokenizer  func(string) []string
}

// NewClassifier returns a pointer to a instance of type Classifier
func NewClassifier() *Classifier {
	return &Classifier{
		Categories: *category.NewCategories(),
	}
}

// Breaks our string into tokens which will be used to train the classifier
func (c *Classifier) tokenizeText(sample string) []string {
	sample = strings.ToLower(sample)
	return strings.Fields(sample)
}

// Returns the tokenizer that we're going to tokenize the text with
func (c *Classifier) getTokenizer() func(string) []string {
	if c.Tokenizer == nil {
		return c.tokenizeText
	}
	return c.Tokenizer
}

// Counts the total occurances of every token in a given string
func (c *Classifier) countTokenOccurances(tokens []string) map[string]int {
	occurances := make(map[string]int)

	for _, token := range tokens {
		if _, ok := occurances[token]; ok {
			occurances[token]++
		} else {
			occurances[token] = 1
		}
	}

	return occurances
}

// Calculates and caches the probabilities that tokens are in (or aren't in) each of our categories
func (c *Classifier) calculateCategoryProbabilities() {
	totalTally := 0.0
	probabilities := make(map[string]float64)

	// Tallying up the tallies for each category
	for name, category := range c.Categories.GetCategories() {
		probabilities[name] = float64(category.GetTally())
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
		cat := c.Categories.GetCategory(name)
		cat.ProbInCat = probability
		cat.ProbNotInCat = 1.0 - probability
	}
}

// Flush empties the categories to remove all values
func (c *Classifier) Flush() {
	c.Categories = *category.NewCategories()
}

// Train takes a text sample and trains a category with it
func (c *Classifier) Train(category string, text string) {
	cat := c.Categories.GetCategory(category)

	tokens := c.getTokenizer()(text)
	occurances := c.countTokenOccurances(tokens)

	for token, count := range occurances {
		cat.TrainToken(token, count)
	}

	c.calculateCategoryProbabilities()
}

// Untrain takes a text sample and untrains a category with it
func (c *Classifier) Untrain(category string, text string) {
	cat := c.Categories.GetCategory(category)

	tokens := c.getTokenizer()(text)
	occurances := c.countTokenOccurances(tokens)

	for token, count := range occurances {
		cat.UntrainToken(token, count)
	}

	c.calculateCategoryProbabilities()
}

// Classify executes bayesian scoring on the sample and returns the highest scoring item
func (c *Classifier) Classify(text string) Classification {
	scores := c.Score(text)
	result := *new(Classification)
	categories := c.Categories.GetCategories()

	// If we had no scores returned we just return the Classification object without a category
	if len(scores) == 0 {
		return result
	}

	for name, score := range scores {
		if score > result.Score {
			result.Category = *categories[name]
			result.Score = score
		}
	}

	return result
}

// Score determines/scores the bayes probability for each of our categories, given a sample of text
func (c *Classifier) Score(text string) map[string]float64 {

	tokens := c.getTokenizer()(text)
	occurances := c.countTokenOccurances(tokens)

	// Map to hold all scores for all categories
	scores := make(map[string]float64)
	categories := c.Categories.GetCategories()
	for key := range categories {
		scores[key] = 0.0
	}

	// Looping through each string token and calculating its bayesian probability
	for word, count := range occurances {
		tokenScores := make(map[string]float64)
		tokenTally := 0.0

		// Getting the tallies of this token from all categories
		for name, category := range categories {
			tokenScores[name] = float64(category.GetTokenCount(word))
			tokenTally += tokenScores[name]
		}

		// If this word had no occurances in any of our categories, we continue
		if tokenTally == 0.0 {
			continue
		}

		for name, tokenScore := range tokenScores {
			probability := c.calculateBayesianProbability(*categories[name], tokenScore, tokenTally)
			scores[name] += float64(count) * probability
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

// Takes a category and some scoring values and returns a bayesian probability
func (c *Classifier) calculateBayesianProbability(category category.Category, tokenScore float64, tokenTally float64) float64 {
	// P that any given token IS in this category
	prc := category.ProbInCat
	// P that any given token is NOT in this category
	prnc := category.ProbNotInCat
	// P that this token is NOT of this category
	tokPrc := (tokenTally - tokenScore) / tokenTally
	// P that this token IS of this category
	tokPrnc := tokenScore / tokenTally

	numerator := tokPrc * prc
	denominator := (tokPrnc * prnc) + numerator

	if denominator != 0.0 {
		return numerator / denominator
	}

	// Default value if the denominator comes out to 0
	return 0.0
}
