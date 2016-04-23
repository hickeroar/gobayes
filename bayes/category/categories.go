package category

// Categories represents all our trained categories and enables us to interact with them.
type Categories struct {
	categories map[string]*Category // Map of category names to categories
}

// NewCategories returns a pointer to a instance of type Categories
func NewCategories() *Categories {
	return &Categories{
		categories: make(map[string]*Category),
	}
}

// AddCategory is responsible for adding a new trainable category
func (cats *Categories) AddCategory(name string) *Category {
	cat := NewCategory(name)

	cats.categories[name] = cat

	return cats.GetCategory(name)
}

// GetCategory returns a specified category
func (cats *Categories) GetCategory(name string) *Category {
	if val, ok := cats.categories[name]; ok {
		return val
	}

	// If we get here, we don't have this category, so we're adding it.
	return cats.AddCategory(name)
}

// DeleteCategory removes a category from the list of categories
func (cats *Categories) DeleteCategory(name string) {
	delete(cats.categories, name)
}

// GetCategories returns the map of all categories
func (cats *Categories) GetCategories() map[string]*Category {
	return cats.categories
}
