package main

func getCategoryList(c *ClassifierAPI) []string {
	categories := c.classifier.Categories.GetCategories()
	var list []string
	for k := range categories {
		list = append(list, k)
	}
	return list
}

// StandardClassifierResponse is a standard response from the api displaying the list of categories and success bool
type StandardClassifierResponse struct {
	Success    bool
	Categories []string
}

// NewStandardClassifierResponse Gets an assembled instance of StandardClassifierResponse
func NewStandardClassifierResponse(c *ClassifierAPI, success bool) *StandardClassifierResponse {
	return &StandardClassifierResponse{
		Success:    success,
		Categories: getCategoryList(c),
	}
}
