package main

import (
	"context"
	"errors"
	"flag"
	"net/http"
	"net/http/httptest"
	"os"
	"sync/atomic"
	"syscall"
	"testing"
	"time"
)

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
	var capturedHandler http.Handler
	newServer = func(_ string, handler http.Handler) httpServer {
		capturedHandler = handler
		return server
	}
	logFatal = func(...interface{}) {}

	flag.CommandLine = flag.NewFlagSet("test", flag.ContinueOnError)
	os.Args = []string{"gobayes.test", "--port", "9999", "--auth-token", "secret-token"}

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

	if capturedHandler == nil {
		t.Fatal("expected handler to be provided to server")
	}

	req := httptest.NewRequest(http.MethodGet, "/info", nil)
	rr := httptest.NewRecorder()
	capturedHandler.ServeHTTP(rr, req)
	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("expected protected endpoint to require auth token, got status %d", rr.Code)
	}
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
