package bayes

import (
	"errors"
	"regexp"
	"sort"
	"strings"
	"sync"
	"unicode"

	"github.com/hickeroar/gobayes/v3/bayes/category"
	"github.com/kljensen/snowball"
	"golang.org/x/text/unicode/norm"
)

// Classification is the result of classifying a text sample.
type Classification struct {
	Category string  `json:"category"`
	Score    float64 `json:"score"`
}

// Classifier trains text categories and classifies new text samples.
type Classifier struct {
	categories category.Categories
	Tokenizer  func(string) []string
	mu         sync.RWMutex
}

var categoryNamePattern = regexp.MustCompile(`^[-_A-Za-z0-9]+$`)

// ErrInvalidCategoryName indicates category input did not match the allowed pattern.
var ErrInvalidCategoryName = errors.New("invalid category name")

// NewClassifier returns a new Classifier instance.
func NewClassifier() *Classifier {
	return &Classifier{
		categories: *category.NewCategories(),
	}
}

// tokenizeText normalizes, tokenizes, and stems input text.
func (c *Classifier) tokenizeText(sample string) []string {
	sample = norm.NFKC.String(sample)
	sample = strings.ToLower(sample)
	rawTokens := strings.FieldsFunc(sample, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})

	tokens := make([]string, 0, len(rawTokens))
	for _, token := range rawTokens {
		stemmed, err := snowball.Stem(token, "english", true)
		if err == nil && stemmed != "" {
			token = stemmed
		}
		tokens = append(tokens, token)
	}

	return tokens
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
	c.mu.Lock()
	defer c.mu.Unlock()

	c.categories = *category.NewCategories()
	c.categories.EnsureCategoryProbabilities()
}

// Train updates a category with token counts from a text sample.
func (c *Classifier) Train(category string, text string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if !categoryNamePattern.MatchString(category) {
		return ErrInvalidCategoryName
	}

	cat := c.categories.GetCategory(category)

	tokens := c.getTokenizer()(text)
	occurrences := c.countTokenOccurrences(tokens)

	for token, count := range occurrences {
		_ = cat.TrainToken(token, count)
	}

	c.cleanUpCategory(cat)
	c.categories.EnsureCategoryProbabilities()
	return nil
}

// Untrain removes token counts from a category using a text sample.
func (c *Classifier) Untrain(category string, text string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if !categoryNamePattern.MatchString(category) {
		return ErrInvalidCategoryName
	}

	cat := c.categories.GetCategory(category)

	tokens := c.getTokenizer()(text)
	occurrences := c.countTokenOccurrences(tokens)

	for token, count := range occurrences {
		_ = cat.UntrainToken(token, count)
	}

	c.cleanUpCategory(cat)
	c.categories.EnsureCategoryProbabilities()
	return nil
}

// cleanUpCategory removes an empty category.
func (c *Classifier) cleanUpCategory(cat *category.Category) {
	if cat.GetTally() == 0 {
		c.categories.DeleteCategory(cat.Name())
	}
}

// Classify scores text against all categories and returns the best match.
func (c *Classifier) Classify(text string) Classification {
	c.mu.RLock()
	defer c.mu.RUnlock()

	scores := c.scoreUnlocked(text)
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
	c.mu.RLock()
	defer c.mu.RUnlock()

	return c.scoreUnlocked(text)
}

// Summaries returns a response-oriented snapshot of all category data.
func (c *Classifier) Summaries() map[string]category.CategorySummary {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.categories.Summaries()
}

func (c *Classifier) scoreUnlocked(text string) map[string]float64 {
	tokens := c.getTokenizer()(text)
	occurrences := c.countTokenOccurrences(tokens)

	scores := make(map[string]float64)
	categoryNames := c.categories.Names()
	categoriesByName := make(map[string]*category.Category, len(categoryNames))
	for _, key := range categoryNames {
		cat, _ := c.categories.LookupCategory(key)
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
			cat := categoriesByName[name]
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
