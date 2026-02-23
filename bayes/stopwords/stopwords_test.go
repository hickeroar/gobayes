package stopwords

import (
	"testing"
)

func TestGetAndSupported(t *testing.T) {
	if !Supported("english") {
		t.Fatal("expected english to be supported")
	}
	if !Supported("spanish") {
		t.Fatal("expected spanish to be supported")
	}
	if Supported("unsupported") {
		t.Fatal("expected unsupported language to not be supported")
	}
	if Supported("") {
		t.Fatal("expected empty language to not be supported")
	}

	en := Get("english")
	if en == nil {
		t.Fatal("expected non-nil set for english")
	}
	if _, ok := en["the"]; !ok {
		t.Fatal("expected 'the' in english stopwords")
	}
	if got := Get("nosuchlang"); got != nil {
		t.Fatalf("expected nil for unsupported language, got %v", got)
	}
}
