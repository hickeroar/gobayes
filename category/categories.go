package category

// Categories represents all our trained categories and enables us to interact with them.
type Categories struct {
	categories map[string]Category
}

// AddCategory is responsible for adding a new trainable category
func (cats *Categories) AddCategory(newName string) Category {
	cat := Category{
		name: newName,
	}

	cats.categories[newName] = cat

	return cat
}

// GetCategory returns a specified category
func (cats *Categories) GetCategory(name string) Category {
	if val, ok := cats.categories[name]; ok {
		return val
	}

	// If we get here, we don't have this category, so we're adding it.
	cats.AddCategory(name)
	return cats.GetCategory(name)
}

// GetCategories returns the map of all categories
func (cats *Categories) GetCategories() map[string]Category {
	return cats.categories
}
