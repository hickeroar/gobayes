package main

// CategoryInfo a breakdown of a category's data
type CategoryInfo struct {
	TokenTally   int     // Total tokens in this category
	ProbNotInCat float64 // The probability that any given token is in this category
	ProbInCat    float64 // The probability that any given token is NOT in this category
}

// getCategoryList returns a simple list of all categories
func getCategoryList(c *ClassifierAPI) map[string]*CategoryInfo {
	categories := c.classifier.Categories.GetCategories()
	list := make(map[string]*CategoryInfo)
	for name, cat := range categories {
		catInfo := &CategoryInfo{
			TokenTally:   cat.Tally,
			ProbNotInCat: cat.ProbNotInCat,
			ProbInCat:    cat.ProbInCat,
		}
		list[name] = catInfo
	}
	return list
}

// TrainingClassifierResponse is a standard response from the api displaying the list of categories and success bool
type TrainingClassifierResponse struct {
	Success    bool
	Categories map[string]*CategoryInfo
}

// NewTrainingClassifierResponse Gets an assembled instance of TrainingClassifierResponse
func NewTrainingClassifierResponse(c *ClassifierAPI, success bool) *TrainingClassifierResponse {
	return &TrainingClassifierResponse{
		Success:    success,
		Categories: getCategoryList(c),
	}
}

// InfoClassifierResponse is a standard response from the api displaying the list of categories and success bool
type InfoClassifierResponse struct {
	Categories map[string]*CategoryInfo
}

// NewInfoClassifierResponse Gets an assembled instance of TrainingClassifierResponse
func NewInfoClassifierResponse(c *ClassifierAPI) *InfoClassifierResponse {
	return &InfoClassifierResponse{
		Categories: getCategoryList(c),
	}
}
