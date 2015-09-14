package bayes

import (
	"github.com/hickeroar/gobayes/bayes/category"
	"strings"
)

// Classifier is responsible for classifying text samples
type Classifier struct {
	Categories category.Categories
	Tokenizer  func(string) []string
}

// NewClassifier returns a pointer to a instance of type Classifier
func NewClassifier() Classifier {
	return Classifier{
		Categories: category.NewCategories(),
	}
}

// Breaks our string into tokens which will be used to train the classifier
func (c *Classifier) tokenizeText(sentence string) []string {
	return strings.Fields(sentence)
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

	for name, category := range c.Categories.GetCategories() {
		probabilities[name] = float64(category.GetTally())
		totalTally += probabilities[name]
	}

	if totalTally > 0.0 {
		for name, count := range probabilities {
			probabilities[name] = count / totalTally
		}
	}

	for name, probability := range probabilities {
		cat := c.Categories.GetCategory(name)
		cat.ProbInCat = probability
		cat.ProbNotInCat = 1.0 - probability
	}
}

// Flush empties the categories to remove all values
func (c *Classifier) Flush() {
	c.Categories = category.NewCategories()
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
