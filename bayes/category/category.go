package category

// Category represents a single text category
type Category struct {
	Name         string         // Name of this category
	Tokens       map[string]int // Map of tokens to their count
	Tally        int            // Total tokens in this category
	ProbNotInCat float64        // The probability that any given token is in this category
	ProbInCat    float64        // The probability that any given token is NOT in this category
}

// NewCategory returns a pointer to a instance of type Category
func NewCategory(name string) *Category {
	return &Category{
		Name:         name,
		Tokens:       make(map[string]int),
		Tally:        0,
		ProbNotInCat: 0.0,
		ProbInCat:    0.0,
	}
}

// TrainToken trains a specific token on this category
func (cat *Category) TrainToken(word string, count int) {
	// Creating the token if it doesn't exist, otherwise incrementing it
	if _, ok := cat.Tokens[word]; ok {
		cat.Tokens[word] += count
	} else {
		cat.Tokens[word] = count
	}

	cat.Tally += count
}

// UntrainToken untrains a specific token on this category
func (cat *Category) UntrainToken(word string, count int) {
	// If the token isn't defined we just return
	if _, ok := cat.Tokens[word]; ok {
		return
	}

	// if we're removing equal or more counts than we have, we kill the token
	if count >= cat.Tokens[word] {
		cat.Tally -= cat.Tokens[word]
		delete(cat.Tokens, word)
	} else {
		cat.Tokens[word] -= count
		cat.Tally -= count
	}
}

// GetTokenCount returns at tokens count from this category
func (cat *Category) GetTokenCount(word string) int {
	if val, ok := cat.Tokens[word]; ok {
		return val
	}
	return 0
}

// GetTally returns the total of all tokens for this category
func (cat *Category) GetTally() int {
	return cat.Tally
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
