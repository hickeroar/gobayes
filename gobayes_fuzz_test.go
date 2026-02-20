package main

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func FuzzCategoryFromPath(f *testing.F) {
	f.Add("/train/spam", "/train/")
	f.Add("/train/spam123", "/train/")
	f.Add("/train/spam-v2", "/train/")
	f.Add("/train/spam_v2", "/train/")
	f.Add("/untrain/ham", "/untrain/")
	f.Add("/train/", "/train/")
	f.Add("/train/with/slash", "/train/")

	f.Fuzz(func(t *testing.T, path string, prefix string) {
		category, ok := categoryFromPath(path, prefix)
		if ok {
			if !categoryPathPattern.MatchString(category) {
				t.Fatalf("category %q accepted but does not match pattern", category)
			}
			if strings.Contains(category, "/") {
				t.Fatalf("category %q accepted but still contains slash", category)
			}
		}
	})
}

func FuzzClassifyHandlerBody(f *testing.F) {
	f.Add(http.MethodPost, "buy now")
	f.Add(http.MethodPost, string(bytes.Repeat([]byte("a"), maxRequestBodyBytes+1)))
	f.Add(http.MethodGet, "ignored")

	f.Fuzz(func(t *testing.T, method string, body string) {
		_, mux := newTestServer()
		req, err := http.NewRequest(method, "http://example.com/classify", strings.NewReader(body))
		if err != nil {
			return
		}
		rr := httptest.NewRecorder()
		mux.ServeHTTP(rr, req)

		switch rr.Code {
		case http.StatusOK, http.StatusMethodNotAllowed, http.StatusBadRequest, http.StatusRequestEntityTooLarge:
			// expected statuses
		default:
			t.Fatalf("unexpected status code %d for method=%q bodyLen=%d", rr.Code, method, len(body))
		}

		if rr.Code != http.StatusOK {
			assertJSONErrorShape(t, rr)
		}
	})
}
