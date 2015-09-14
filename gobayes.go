package main

import (
	"fmt"
	"github.com/hickeroar/gobayes/bayes"
)

func main() {
	class := bayes.NewClassifier()

	class.Categories.AddCategory("foo")
	class.Categories.AddCategory("baz")
	class.Categories.AddCategory("baz")
	class.Train("foo", "the boy's bomb is stupid, round, and most likely a dud")
	class.Train("bar", "the quick brown fox is the bomb")
	class.Train("baz", "your mother is not the bomb")

	categories := class.Categories.GetCategories()

	for name, cat := range categories {
		fmt.Println(name)
		fmt.Println("Tally:", cat.GetTally())
		fmt.Println(cat.ProbInCat)
		fmt.Println(cat.ProbNotInCat)
	}
}
