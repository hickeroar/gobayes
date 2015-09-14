package main

import "fmt"

func main() {
	class := NewClassifier()

	class.Categories.AddCategory("bleh")
	class.Categories.AddCategory("gleh")

	categories := class.Categories.GetCategories()

	for name, cat := range categories {
		fmt.Println(name)
		fmt.Println("Tally:", cat.GetTally())
	}

}
