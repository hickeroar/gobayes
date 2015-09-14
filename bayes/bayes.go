package bayes

import (
	"github.com/hickeroar/gobayes/category"
	"strings"
)

// Classifier is responsible for classifying text samples
type Classifier struct {
	Categories *category.Categories
	Tokenizer  func(string) []string
}

// NewClassifier returns a pointer to a instance of type Classifier
func NewClassifier() *Classifier {
	return &Classifier{
		Categories: category.NewCategories(),
	}
}

func (c *Classifier) tokenizeText(sentence string) []string {
	return strings.Fields(sentence)
}

// Flush empties the categories to remove all values
func (c *Classifier) Flush() {
	c.Categories = category.NewCategories()
}
