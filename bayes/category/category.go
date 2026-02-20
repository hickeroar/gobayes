package category

import "errors"

// ErrInvalidTokenCount indicates token mutation was requested with a non-positive count.
var ErrInvalidTokenCount = errors.New("count must be greater than zero")

// Category stores token and probability data for one classification category.
type Category struct {
	name         string         // Name of this category
	tokens       map[string]int // Map of tokens to their count
	tally        int            // Total tokens in this category
	probNotInCat float64        // Probability that an arbitrary token is not in this category
	probInCat    float64        // Probability that an arbitrary token is in this category
}

// NewCategory returns a new Category with initialized token storage.
func NewCategory(name string) *Category {
	return &Category{
		name:         name,
		tokens:       make(map[string]int),
		tally:        0,
		probNotInCat: 0.0,
		probInCat:    0.0,
	}
}

// TrainToken adds count occurrences of word to the category.
func (cat *Category) TrainToken(word string, count int) error {
	if count <= 0 {
		return ErrInvalidTokenCount
	}

	cat.tokens[word] += count

	cat.tally += count
	return nil
}

// UntrainToken removes count occurrences of word from the category.
func (cat *Category) UntrainToken(word string, count int) error {
	if count <= 0 {
		return ErrInvalidTokenCount
	}

	curCount, keyExists := cat.tokens[word]

	if keyExists {
		// if we're removing equal or more counts than we have, we kill the token
		if count >= curCount {
			cat.tally -= cat.tokens[word]
			delete(cat.tokens, word)
		} else {
			cat.tokens[word] -= count
			cat.tally -= count
		}
	}
	return nil
}

// Name returns the category name.
func (cat Category) Name() string {
	return cat.name
}

// GetTokenCount returns the number of times word appears in the category.
func (cat Category) GetTokenCount(word string) int {
	if val, ok := cat.tokens[word]; ok {
		return val
	}
	return 0
}

// GetTally returns the total trained token count for this category.
func (cat Category) GetTally() int {
	return cat.tally
}

// GetProbInCat returns the prior probability for the category.
func (cat Category) GetProbInCat() float64 {
	return cat.probInCat
}

// GetProbNotInCat returns the complement prior probability for the category.
func (cat Category) GetProbNotInCat() float64 {
	return cat.probNotInCat
}

func (cat *Category) setProbabilities(probInCat float64, probNotInCat float64) {
	cat.probInCat = probInCat
	cat.probNotInCat = probNotInCat
}
