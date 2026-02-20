package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

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

func TestTrainAndUntrainHandlers(t *testing.T) {
	_, mux := newTestServer()

	trainReq := httptest.NewRequest(http.MethodPost, "/train/spam", strings.NewReader("buy now now"))
	trainRR := httptest.NewRecorder()
	mux.ServeHTTP(trainRR, trainReq)
	if trainRR.Code != http.StatusOK {
		t.Fatalf("unexpected train status: got %d, want %d", trainRR.Code, http.StatusOK)
	}

	untrainReq := httptest.NewRequest(http.MethodPost, "/untrain/spam", strings.NewReader("buy"))
	untrainRR := httptest.NewRecorder()
	mux.ServeHTTP(untrainRR, untrainReq)
	if untrainRR.Code != http.StatusOK {
		t.Fatalf("unexpected untrain status: got %d, want %d", untrainRR.Code, http.StatusOK)
	}
}

func TestUntrainInvalidCategoryRoute(t *testing.T) {
	_, mux := newTestServer()
	req := httptest.NewRequest(http.MethodPost, "/untrain/spam!", strings.NewReader("text"))
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)
	if rr.Code != http.StatusNotFound {
		t.Fatalf("unexpected status: got %d, want %d", rr.Code, http.StatusNotFound)
	}
	assertJSONErrorShape(t, rr)
}

func TestScoreHandlerBadBody(t *testing.T) {
	_, mux := newTestServer()
	req := httptest.NewRequest(http.MethodPost, "/score", nil)
	req.Body = io.NopCloser(errReader{})
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("unexpected status: got %d, want %d", rr.Code, http.StatusBadRequest)
	}
	assertJSONErrorShape(t, rr)
}

func TestTrainHandlerBadBody(t *testing.T) {
	_, mux := newTestServer()
	req := httptest.NewRequest(http.MethodPost, "/train/spam", nil)
	req.Body = io.NopCloser(errReader{})
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("unexpected status: got %d, want %d", rr.Code, http.StatusBadRequest)
	}
	assertJSONErrorShape(t, rr)
}

func TestUntrainHandlerBadBody(t *testing.T) {
	_, mux := newTestServer()
	req := httptest.NewRequest(http.MethodPost, "/untrain/spam", nil)
	req.Body = io.NopCloser(errReader{})
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("unexpected status: got %d, want %d", rr.Code, http.StatusBadRequest)
	}
	assertJSONErrorShape(t, rr)
}

func TestTrainHandlerMethodNotAllowed(t *testing.T) {
	_, mux := newTestServer()
	req := httptest.NewRequest(http.MethodGet, "/train/spam", nil)
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)
	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("unexpected status: got %d, want %d", rr.Code, http.StatusMethodNotAllowed)
	}
	assertJSONErrorShape(t, rr)
}

func TestHealthHandlerMethodNotAllowed(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/healthz", nil)
	rr := httptest.NewRecorder()
	HealthHandler(rr, req)
	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("unexpected status: got %d, want %d", rr.Code, http.StatusMethodNotAllowed)
	}
	assertJSONErrorShape(t, rr)
}

func TestReadyHandlerMethodNotAllowed(t *testing.T) {
	api, _ := newTestServer()
	req := httptest.NewRequest(http.MethodPost, "/readyz", nil)
	rr := httptest.NewRecorder()
	api.ReadyHandler(rr, req)
	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("unexpected status: got %d, want %d", rr.Code, http.StatusMethodNotAllowed)
	}
	assertJSONErrorShape(t, rr)
}

type errReader struct{}

func (errReader) Read([]byte) (int, error) {
	return 0, errors.New("read failed")
}

type failWriteRecorder struct {
	header http.Header
	status int
}

func (f *failWriteRecorder) Header() http.Header {
	return f.header
}

func (f *failWriteRecorder) WriteHeader(statusCode int) {
	f.status = statusCode
}

func (f *failWriteRecorder) Write([]byte) (int, error) {
	return 0, errors.New("write failed")
}

func TestWriteJSONMarshalAndWriteErrors(t *testing.T) {
	rr := httptest.NewRecorder()
	writeJSON(rr, http.StatusOK, map[string]interface{}{"bad": func() {}})
	if rr.Code != http.StatusInternalServerError {
		t.Fatalf("expected internal server error for marshal failure, got %d", rr.Code)
	}

	failing := &failWriteRecorder{header: make(http.Header)}
	writeJSON(failing, http.StatusOK, map[string]string{"ok": "ok"})
	if failing.status != http.StatusOK {
		t.Fatalf("expected status to be set before write failure, got %d", failing.status)
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

type fakeServer struct {
	listenErr   error
	shutdownErr error
	listened    atomic.Bool
}

func (f *fakeServer) ListenAndServe() error {
	f.listened.Store(true)
	return f.listenErr
}

func (f *fakeServer) Shutdown(context.Context) error {
	return f.shutdownErr
}

func TestRunMainSuccessPath(t *testing.T) {
	oldRunMain := runMain
	oldMakeSignal := makeSignalChannel
	oldNotify := notifySignals
	oldNewServer := newServer
	oldLogFatal := logFatal
	oldFlagCommandLine := flag.CommandLine
	oldArgs := os.Args
	defer func() {
		runMain = oldRunMain
		makeSignalChannel = oldMakeSignal
		notifySignals = oldNotify
		newServer = oldNewServer
		logFatal = oldLogFatal
		flag.CommandLine = oldFlagCommandLine
		os.Args = oldArgs
	}()

	sigCh := make(chan os.Signal, 1)
	makeSignalChannel = func() chan os.Signal { return sigCh }
	notifySignals = func(chan<- os.Signal, ...os.Signal) {}

	server := &fakeServer{listenErr: http.ErrServerClosed}
	newServer = func(string, http.Handler) httpServer { return server }
	logFatal = func(...interface{}) {}

	flag.CommandLine = flag.NewFlagSet("test", flag.ContinueOnError)
	os.Args = []string{"gobayes.test", "-port", "9999"}

	done := make(chan error, 1)
	go func() {
		done <- runMain()
	}()

	sigCh <- syscall.SIGTERM

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("expected nil runMain error, got %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for runMain to exit")
	}

	_ = server.listened.Load()
}

func TestMainHandlesRunError(t *testing.T) {
	oldRunMain := runMain
	oldLogFatal := logFatal
	defer func() {
		runMain = oldRunMain
		logFatal = oldLogFatal
	}()

	expectedErr := errors.New("boom")
	runMain = func() error { return expectedErr }

	called := false
	logFatal = func(v ...interface{}) {
		called = true
		if len(v) != 1 {
			t.Fatalf("unexpected fatal args: %v", v)
		}
		if !errors.Is(v[0].(error), expectedErr) {
			t.Fatalf("unexpected fatal error: %v", v[0])
		}
	}

	main()
	if !called {
		t.Fatal("expected main to call logFatal on error")
	}
}
