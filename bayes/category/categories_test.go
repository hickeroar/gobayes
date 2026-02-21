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

func TestNamesReturnsCategoryNames(t *testing.T) {
	cats := NewCategories()
	cats.AddCategory("spam")
	cats.AddCategory("ham")

	names := cats.Names()
	if len(names) != 2 {
		t.Fatalf("expected 2 names, got %d", len(names))
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

func TestExportAndReplaceStates(t *testing.T) {
	original := NewCategories()
	spam := original.GetCategory("spam")
	if err := spam.TrainToken("buy", 2); err != nil {
		t.Fatalf("unexpected error training token: %v", err)
	}
	if err := spam.TrainToken("now", 1); err != nil {
		t.Fatalf("unexpected error training token: %v", err)
	}

	states := original.ExportStates()

	restored := NewCategories()
	if err := restored.ReplaceStates(states); err != nil {
		t.Fatalf("replace states failed: %v", err)
	}

	cat, ok := restored.LookupCategory("spam")
	if !ok {
		t.Fatal("expected spam category after restore")
	}
	if got := cat.GetTokenCount("buy"); got != 2 {
		t.Fatalf("unexpected buy count: got %d want 2", got)
	}
	if got := cat.GetTally(); got != 3 {
		t.Fatalf("unexpected tally: got %d want 3", got)
	}
}

func TestReplaceStatesRejectsInvalidState(t *testing.T) {
	cats := NewCategories()
	err := cats.ReplaceStates(map[string]PersistedCategory{
		"spam": {Tokens: map[string]int{"buy": 2}, Tally: 1},
	})
	if err == nil {
		t.Fatal("expected error for tally mismatch")
	}
}

func TestReplaceStatesRejectsInvalidTokenCount(t *testing.T) {
	cats := NewCategories()
	err := cats.ReplaceStates(map[string]PersistedCategory{
		"spam": {Tokens: map[string]int{"buy": 0}, Tally: 0},
	})
	if err == nil {
		t.Fatal("expected error for invalid token count")
	}
}

func TestEnsureCategoryProbabilitiesNoOpWhenClean(t *testing.T) {
	cats := NewCategories()
	spam := cats.GetCategory("spam")
	if err := spam.TrainToken("buy", 2); err != nil {
		t.Fatalf("unexpected train error: %v", err)
	}
	cats.EnsureCategoryProbabilities()
	before := cats.Summaries()["spam"].ProbInCat

	cats.EnsureCategoryProbabilities()
	after := cats.Summaries()["spam"].ProbInCat

	if before != after {
		t.Fatal("expected probabilities unchanged when clean")
	}
}

func TestMarkProbabilitiesDirtyForcesRecalc(t *testing.T) {
	cats := NewCategories()
	spam := cats.GetCategory("spam")
	if err := spam.TrainToken("buy", 2); err != nil {
		t.Fatalf("unexpected train error: %v", err)
	}
	cats.EnsureCategoryProbabilities()
	before := cats.Summaries()["spam"].ProbInCat

	ham := cats.GetCategory("ham")
	if err := ham.TrainToken("team", 2); err != nil {
		t.Fatalf("unexpected train error: %v", err)
	}
	cats.MarkProbabilitiesDirty()
	cats.EnsureCategoryProbabilities()
	after := cats.Summaries()["spam"].ProbInCat

	if before == after {
		t.Fatal("expected probability to change after marking dirty and recalculating")
	}
}
