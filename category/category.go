package category

// Category represents a single text category
type Category struct {
	name   string
	tokens map[string]int
	tally  int
}

// TrainToken trains a specific token on this category
func (cat *Category) TrainToken(word string, count int) {
	// Creating the token if it doesn't exist, otherwise incrementing it
	if _, ok := cat.tokens[word]; ok {
		cat.tokens[word] += count
	} else {
		cat.tokens[word] = count
	}

	cat.tally += count
}

// UntrainToken untrains a specific token on this category
func (cat *Category) UntrainToken(word string, count int) {
	// If the token isn't defined we just return
	if _, ok := cat.tokens[word]; ok {
		return
	}

	// if we're removing equal or more counts than we have, we kill the token
	if count >= cat.tokens[word] {
		cat.tally -= cat.tokens[word]
		delete(cat.tokens, word)
	} else {
		cat.tokens[word] -= count
		cat.tally -= count
	}
}

// GetTokenCount returns at tokens count from this category
func (cat *Category) GetTokenCount(word string) int {
	if val, ok := cat.tokens[word]; ok {
		return val
	}
	return 0
}

// GetTally returns the total of all tokens for this category
func (cat *Category) GetTally() int {
	return cat.tally
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
