package category

import "testing"

func TestTrainTokenCreatesAndIncrements(t *testing.T) {
	cat := NewCategory("spam")

	if err := cat.TrainToken("buy", 2); err != nil {
		t.Fatalf("unexpected error training token: %v", err)
	}
	if err := cat.TrainToken("buy", 3); err != nil {
		t.Fatalf("unexpected error training token: %v", err)
	}
	if err := cat.TrainToken("now", 1); err != nil {
		t.Fatalf("unexpected error training token: %v", err)
	}

	if got := cat.GetTokenCount("buy"); got != 5 {
		t.Fatalf("unexpected buy count: got %d, want %d", got, 5)
	}
	if got := cat.GetTokenCount("now"); got != 1 {
		t.Fatalf("unexpected now count: got %d, want %d", got, 1)
	}
	if got := cat.GetTally(); got != 6 {
		t.Fatalf("unexpected tally: got %d, want %d", got, 6)
	}
}

func TestUntrainTokenDecrementsAndDeletes(t *testing.T) {
	cat := NewCategory("spam")
	if err := cat.TrainToken("buy", 5); err != nil {
		t.Fatalf("unexpected error training token: %v", err)
	}
	if err := cat.TrainToken("now", 1); err != nil {
		t.Fatalf("unexpected error training token: %v", err)
	}

	if err := cat.UntrainToken("buy", 2); err != nil {
		t.Fatalf("unexpected error untraining token: %v", err)
	}
	if got := cat.GetTokenCount("buy"); got != 3 {
		t.Fatalf("unexpected buy count after partial untrain: got %d, want %d", got, 3)
	}
	if got := cat.GetTally(); got != 4 {
		t.Fatalf("unexpected tally after partial untrain: got %d, want %d", got, 4)
	}

	if err := cat.UntrainToken("buy", 3); err != nil {
		t.Fatalf("unexpected error untraining token: %v", err)
	}
	if got := cat.GetTokenCount("buy"); got != 0 {
		t.Fatalf("expected buy token to be removed, got count %d", got)
	}
	if got := cat.GetTally(); got != 1 {
		t.Fatalf("unexpected tally after removing buy token: got %d, want %d", got, 1)
	}
}

func TestUntrainTokenNoOpForMissingToken(t *testing.T) {
	cat := NewCategory("ham")
	if err := cat.TrainToken("hello", 2); err != nil {
		t.Fatalf("unexpected error training token: %v", err)
	}

	if err := cat.UntrainToken("missing", 5); err != nil {
		t.Fatalf("unexpected error untraining missing token: %v", err)
	}

	if got := cat.GetTokenCount("hello"); got != 2 {
		t.Fatalf("existing token should remain unchanged: got %d, want %d", got, 2)
	}
	if got := cat.GetTally(); got != 2 {
		t.Fatalf("tally should remain unchanged: got %d, want %d", got, 2)
	}
}

func TestInvalidCountsReturnError(t *testing.T) {
	cat := NewCategory("ham")
	if err := cat.TrainToken("hello", 2); err != nil {
		t.Fatalf("unexpected error training token: %v", err)
	}

	if err := cat.TrainToken("hello", 0); err == nil {
		t.Fatal("expected error for zero training count")
	}
	if err := cat.TrainToken("hello", -3); err == nil {
		t.Fatal("expected error for negative training count")
	}
	if err := cat.UntrainToken("hello", 0); err == nil {
		t.Fatal("expected error for zero untraining count")
	}
	if err := cat.UntrainToken("hello", -3); err == nil {
		t.Fatal("expected error for negative untraining count")
	}

	if got := cat.GetTokenCount("hello"); got != 2 {
		t.Fatalf("expected count unchanged after invalid operations: got %d, want %d", got, 2)
	}
	if got := cat.GetTally(); got != 2 {
		t.Fatalf("expected tally unchanged after invalid operations: got %d, want %d", got, 2)
	}
}
