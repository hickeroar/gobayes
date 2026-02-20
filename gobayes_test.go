package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/hickeroar/gobayes/v2/bayes"
)

func assertJSONContentType(t *testing.T, rr *httptest.ResponseRecorder) {
	t.Helper()
	contentType := rr.Header().Get("Content-Type")
	if !strings.HasPrefix(contentType, "application/json") {
		t.Fatalf("expected application/json content type, got %q", contentType)
	}
}

func assertJSONErrorShape(t *testing.T, rr *httptest.ResponseRecorder) {
	t.Helper()
	assertJSONContentType(t, rr)
	var payload map[string]string
	if err := json.Unmarshal(rr.Body.Bytes(), &payload); err != nil {
		t.Fatalf("expected JSON error payload: %v", err)
	}
	if payload["error"] == "" {
		t.Fatalf("expected non-empty error field, got payload=%v", payload)
	}
}

func newTestServer() (*ClassifierAPI, *http.ServeMux) {
	api := &ClassifierAPI{
		classifier: *bayes.NewClassifier(),
	}
	api.ready.Store(true)
	mux := http.NewServeMux()
	api.RegisterRoutes(mux)
	return api, mux
}

func TestClassifyMethodNotAllowed(t *testing.T) {
	_, mux := newTestServer()
	req := httptest.NewRequest(http.MethodGet, "/classify", nil)
	rr := httptest.NewRecorder()

	mux.ServeHTTP(rr, req)

	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("unexpected status: got %d, want %d", rr.Code, http.StatusMethodNotAllowed)
	}
	if allow := rr.Header().Get("Allow"); allow != http.MethodPost {
		t.Fatalf("unexpected Allow header: got %q, want %q", allow, http.MethodPost)
	}
	assertJSONErrorShape(t, rr)
}

func TestTrainInfoFlushLifecycle(t *testing.T) {
	_, mux := newTestServer()

	trainReq := httptest.NewRequest(http.MethodPost, "/train/spam", strings.NewReader("buy now"))
	trainRR := httptest.NewRecorder()
	mux.ServeHTTP(trainRR, trainReq)
	if trainRR.Code != http.StatusOK {
		t.Fatalf("unexpected train status: got %d, want %d", trainRR.Code, http.StatusOK)
	}

	infoReq := httptest.NewRequest(http.MethodGet, "/info", nil)
	infoRR := httptest.NewRecorder()
	mux.ServeHTTP(infoRR, infoReq)
	if infoRR.Code != http.StatusOK {
		t.Fatalf("unexpected info status: got %d, want %d", infoRR.Code, http.StatusOK)
	}

	var infoResp struct {
		Categories map[string]json.RawMessage
	}
	if err := json.Unmarshal(infoRR.Body.Bytes(), &infoResp); err != nil {
		t.Fatalf("failed to unmarshal info response: %v", err)
	}
	if _, ok := infoResp.Categories["spam"]; !ok {
		t.Fatal("expected spam category in info response")
	}

	flushReq := httptest.NewRequest(http.MethodPost, "/flush", nil)
	flushRR := httptest.NewRecorder()
	mux.ServeHTTP(flushRR, flushReq)
	if flushRR.Code != http.StatusOK {
		t.Fatalf("unexpected flush status: got %d, want %d", flushRR.Code, http.StatusOK)
	}

	var flushResp struct {
		Success    bool
		Categories map[string]json.RawMessage
	}
	if err := json.Unmarshal(flushRR.Body.Bytes(), &flushResp); err != nil {
		t.Fatalf("failed to unmarshal flush response: %v", err)
	}
	if !flushResp.Success {
		t.Fatal("expected flush success=true")
	}
	if len(flushResp.Categories) != 0 {
		t.Fatalf("expected no categories after flush, got %d", len(flushResp.Categories))
	}
}

func TestInvalidCategoryRoute(t *testing.T) {
	_, mux := newTestServer()
	req := httptest.NewRequest(http.MethodPost, "/train/spam!", strings.NewReader("text"))
	rr := httptest.NewRecorder()

	mux.ServeHTTP(rr, req)

	if rr.Code != http.StatusNotFound {
		t.Fatalf("unexpected status: got %d, want %d", rr.Code, http.StatusNotFound)
	}
}

func TestOversizedBodyRejected(t *testing.T) {
	_, mux := newTestServer()
	oversized := bytes.Repeat([]byte("a"), maxRequestBodyBytes+1)
	req := httptest.NewRequest(http.MethodPost, "/classify", bytes.NewReader(oversized))
	rr := httptest.NewRecorder()

	mux.ServeHTTP(rr, req)

	if rr.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("unexpected status: got %d, want %d", rr.Code, http.StatusRequestEntityTooLarge)
	}
	assertJSONErrorShape(t, rr)
}

func TestHealthAndReadyEndpoints(t *testing.T) {
	api, mux := newTestServer()
	api.ready.Store(true)

	for _, path := range []string{"/healthz", "/readyz"} {
		req := httptest.NewRequest(http.MethodGet, path, nil)
		rr := httptest.NewRecorder()
		mux.ServeHTTP(rr, req)
		if rr.Code != http.StatusOK {
			t.Fatalf("unexpected status for %s: got %d, want %d", path, rr.Code, http.StatusOK)
		}
		assertJSONContentType(t, rr)
	}
}

func TestReadyEndpointNotReady(t *testing.T) {
	api, mux := newTestServer()
	api.ready.Store(false)

	req := httptest.NewRequest(http.MethodGet, "/readyz", nil)
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)

	if rr.Code != http.StatusServiceUnavailable {
		t.Fatalf("unexpected status for /readyz: got %d, want %d", rr.Code, http.StatusServiceUnavailable)
	}
	assertJSONContentType(t, rr)
}

func TestConcurrentRequests(t *testing.T) {
	_, mux := newTestServer()

	var wg sync.WaitGroup
	for i := 0; i < 30; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			trainReq := httptest.NewRequest(http.MethodPost, "/train/spam", strings.NewReader("buy now buy now"))
			trainRR := httptest.NewRecorder()
			mux.ServeHTTP(trainRR, trainReq)
			if trainRR.Code != http.StatusOK {
				t.Errorf("unexpected train status: got %d", trainRR.Code)
			}

			classifyReq := httptest.NewRequest(http.MethodPost, "/classify", strings.NewReader("buy now"))
			classifyRR := httptest.NewRecorder()
			mux.ServeHTTP(classifyRR, classifyReq)
			if classifyRR.Code != http.StatusOK {
				t.Errorf("unexpected classify status: got %d", classifyRR.Code)
			}

			scoreReq := httptest.NewRequest(http.MethodPost, "/score", strings.NewReader("buy now"))
			scoreRR := httptest.NewRecorder()
			mux.ServeHTTP(scoreRR, scoreReq)
			if scoreRR.Code != http.StatusOK {
				t.Errorf("unexpected score status: got %d", scoreRR.Code)
			}
		}()
	}
	wg.Wait()
}

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

func TestAPIContractMatrix(t *testing.T) {
	type testCase struct {
		name        string
		method      string
		path        string
		body        []byte
		status      int
		allowHeader string
		expectError bool
	}

	oversized := bytes.Repeat([]byte("a"), maxRequestBodyBytes+1)
	tests := []testCase{
		{name: "info get ok", method: http.MethodGet, path: "/info", status: http.StatusOK},
		{name: "info wrong method", method: http.MethodPost, path: "/info", status: http.StatusMethodNotAllowed, allowHeader: http.MethodGet, expectError: true},
		{name: "train accepts broadened category", method: http.MethodPost, path: "/train/spam_v2", body: []byte("buy now"), status: http.StatusOK},
		{name: "train rejects invalid category", method: http.MethodPost, path: "/train/spam!", body: []byte("buy now"), status: http.StatusNotFound, expectError: true},
		{name: "untrain wrong method", method: http.MethodGet, path: "/untrain/spam", status: http.StatusMethodNotAllowed, allowHeader: http.MethodPost, expectError: true},
		{name: "classify wrong method", method: http.MethodGet, path: "/classify", status: http.StatusMethodNotAllowed, allowHeader: http.MethodPost, expectError: true},
		{name: "classify oversized body", method: http.MethodPost, path: "/classify", body: oversized, status: http.StatusRequestEntityTooLarge, expectError: true},
		{name: "score wrong method", method: http.MethodGet, path: "/score", status: http.StatusMethodNotAllowed, allowHeader: http.MethodPost, expectError: true},
		{name: "flush wrong method", method: http.MethodGet, path: "/flush", status: http.StatusMethodNotAllowed, allowHeader: http.MethodPost, expectError: true},
		{name: "healthz get ok", method: http.MethodGet, path: "/healthz", status: http.StatusOK},
		{name: "readyz get ok", method: http.MethodGet, path: "/readyz", status: http.StatusOK},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, mux := newTestServer()
			req := httptest.NewRequest(tc.method, tc.path, bytes.NewReader(tc.body))
			rr := httptest.NewRecorder()
			mux.ServeHTTP(rr, req)

			if rr.Code != tc.status {
				t.Fatalf("unexpected status: got %d want %d", rr.Code, tc.status)
			}

			assertJSONContentType(t, rr)

			if tc.allowHeader != "" {
				if allow := rr.Header().Get("Allow"); allow != tc.allowHeader {
					t.Fatalf("unexpected Allow header: got %q want %q", allow, tc.allowHeader)
				}
			}
			if tc.expectError {
				assertJSONErrorShape(t, rr)
			}
		})
	}
}
