package category

import "testing"

func TestAddCategoryCreatesAndReturnsCategory(t *testing.T) {
	cats := NewCategories()
	cat := cats.AddCategory("spam")

	if cat == nil {
		t.Fatal("expected non-nil category")
	}
	if cat.Name() != "spam" {
		t.Fatalf("unexpected category name: got %q, want %q", cat.Name(), "spam")
	}

	all := cats.Summaries()
	if _, ok := all["spam"]; !ok {
		t.Fatal("expected spam category to exist in map")
	}
}

func TestGetCategoryReturnsExistingAndCreatesMissing(t *testing.T) {
	cats := NewCategories()

	first := cats.GetCategory("ham")
	second := cats.GetCategory("ham")

	if first != second {
		t.Fatal("expected GetCategory to return same pointer for existing category")
	}

	missing := cats.GetCategory("spam")
	if missing == nil || missing.Name() != "spam" {
		t.Fatal("expected missing category to be lazily created")
	}
}

func TestDeleteCategoryRemovesCategory(t *testing.T) {
	cats := NewCategories()
	cats.AddCategory("spam")
	cats.AddCategory("ham")

	cats.DeleteCategory("spam")

	all := cats.Summaries()
	if _, ok := all["spam"]; ok {
		t.Fatal("expected spam category to be deleted")
	}
	if _, ok := all["ham"]; !ok {
		t.Fatal("expected ham category to remain")
	}
}

func TestSummariesReturnsValueSnapshot(t *testing.T) {
	cats := NewCategories()
	created := cats.AddCategory("spam")
	if err := created.TrainToken("buy", 2); err != nil {
		t.Fatalf("unexpected error training token: %v", err)
	}

	snapshot := cats.Summaries()
	entry := snapshot["spam"]
	entry.TokenTally = 999
	snapshot["spam"] = entry
	delete(snapshot, "spam")

	real := cats.GetCategory("spam")
	if got := real.GetTokenCount("buy"); got != 2 {
		t.Fatalf("expected internal state unchanged by snapshot mutation: got %d, want %d", got, 2)
	}
	if _, ok := cats.Summaries()["spam"]; !ok {
		t.Fatal("expected category to remain after snapshot map deletion")
	}
}
