package main

// CategoryInfo describes summary data for a trained category.
type CategoryInfo struct {
	TokenTally   int     `json:"TokenTally"`   // Total tokens in this category
	ProbNotInCat float64 `json:"ProbNotInCat"` // Probability that an arbitrary token is not in this category
	ProbInCat    float64 `json:"ProbInCat"`    // Probability that an arbitrary token is in this category
}

// getCategoryList returns a summary view of all categories.
func getCategoryList(c *ClassifierAPI) map[string]*CategoryInfo {
	categories := c.classifier.Summaries()
	list := make(map[string]*CategoryInfo)
	for name, cat := range categories {
		catInfo := &CategoryInfo{
			TokenTally:   cat.TokenTally,
			ProbNotInCat: cat.ProbNotInCat,
			ProbInCat:    cat.ProbInCat,
		}
		list[name] = catInfo
	}
	return list
}

// TrainingClassifierResponse is returned by train, untrain, and flush endpoints.
type TrainingClassifierResponse struct {
	Success    bool                     `json:"Success"`
	Categories map[string]*CategoryInfo `json:"Categories"`
}

// NewTrainingClassifierResponse builds a TrainingClassifierResponse.
func NewTrainingClassifierResponse(c *ClassifierAPI, success bool) *TrainingClassifierResponse {
	return &TrainingClassifierResponse{
		Success:    success,
		Categories: getCategoryList(c),
	}
}

// InfoClassifierResponse is returned by the info endpoint.
type InfoClassifierResponse struct {
	Categories map[string]*CategoryInfo `json:"Categories"`
}

// NewInfoClassifierResponse builds an InfoClassifierResponse.
func NewInfoClassifierResponse(c *ClassifierAPI) *InfoClassifierResponse {
	return &InfoClassifierResponse{
		Categories: getCategoryList(c),
	}
}
