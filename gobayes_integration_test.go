//go:build integration

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"path/filepath"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

type integrationServer struct {
	baseURL string
	client  *http.Client
	cmd     *exec.Cmd
	waitCh  chan error
	output  *bytes.Buffer
}

func startIntegrationServer(t *testing.T, authToken string) *integrationServer {
	t.Helper()

	port := reservePort(t)
	binPath := filepath.Join(t.TempDir(), "gobayes-integration")

	buildCmd := exec.Command("go", "build", "-o", binPath, ".")
	var buildOutput bytes.Buffer
	buildCmd.Stdout = &buildOutput
	buildCmd.Stderr = &buildOutput
	if err := buildCmd.Run(); err != nil {
		t.Fatalf("build integration binary: %v\nbuild output:\n%s", err, buildOutput.String())
	}

	args := []string{"--port", strconv.Itoa(port)}
	if authToken != "" {
		args = append(args, "--auth-token", authToken)
	}

	cmd := exec.Command(binPath, args...)
	var output bytes.Buffer
	cmd.Stdout = &output
	cmd.Stderr = &output

	if err := cmd.Start(); err != nil {
		t.Fatalf("start server: %v", err)
	}

	server := &integrationServer{
		baseURL: "http://127.0.0.1:" + strconv.Itoa(port),
		client:  &http.Client{Timeout: 2 * time.Second},
		cmd:     cmd,
		waitCh:  make(chan error, 1),
		output:  &output,
	}

	go func() {
		server.waitCh <- cmd.Wait()
	}()

	if err := waitForHealth(server); err != nil {
		server.stop(t)
		t.Fatalf("wait for server readiness: %v\nserver output:\n%s", err, output.String())
	}

	t.Cleanup(func() {
		server.stop(t)
	})

	return server
}

func waitForHealth(server *integrationServer) error {
	deadline := time.Now().Add(8 * time.Second)
	for time.Now().Before(deadline) {
		req, err := http.NewRequest(http.MethodGet, server.baseURL+"/healthz", nil)
		if err != nil {
			return err
		}
		resp, err := server.client.Do(req)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		time.Sleep(100 * time.Millisecond)
	}
	return context.DeadlineExceeded
}

func (s *integrationServer) stop(t *testing.T) {
	t.Helper()
	if s.cmd.Process == nil {
		return
	}

	_ = s.cmd.Process.Signal(syscall.SIGTERM)

	select {
	case <-s.waitCh:
		return
	case <-time.After(5 * time.Second):
		_ = s.cmd.Process.Kill()
		select {
		case <-s.waitCh:
		case <-time.After(2 * time.Second):
			t.Logf("timed out waiting for process after kill")
		}
	}
}

func reservePort(t *testing.T) int {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("reserve port: %v", err)
	}
	defer listener.Close()

	addr, ok := listener.Addr().(*net.TCPAddr)
	if !ok {
		t.Fatalf("unexpected listener addr type: %T", listener.Addr())
	}
	return addr.Port
}

func sendRequest(t *testing.T, server *integrationServer, method, path, body, token string) *http.Response {
	t.Helper()

	req, err := http.NewRequest(method, server.baseURL+path, strings.NewReader(body))
	if err != nil {
		t.Fatalf("build request: %v", err)
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	resp, err := server.client.Do(req)
	if err != nil {
		t.Fatalf("request %s %s failed: %v", method, path, err)
	}
	return resp
}

func decodeJSON[T any](t *testing.T, resp *http.Response) T {
	t.Helper()
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read response body: %v", err)
	}
	var payload T
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("decode JSON failed: %v body=%q", err, string(body))
	}
	return payload
}

func TestIntegrationLifecycleFlow(t *testing.T) {
	server := startIntegrationServer(t, "")

	type trainResp struct {
		Success bool
	}

	resp := sendRequest(t, server, http.MethodPost, "/train/spam", "buy now limited offer", "")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("train spam status: got %d want %d", resp.StatusCode, http.StatusOK)
	}
	_ = decodeJSON[trainResp](t, resp)

	resp = sendRequest(t, server, http.MethodPost, "/train/ham", "team meeting schedule", "")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("train ham status: got %d want %d", resp.StatusCode, http.StatusOK)
	}
	_ = decodeJSON[trainResp](t, resp)

	type classifyResp struct {
		Category string
	}
	resp = sendRequest(t, server, http.MethodPost, "/classify", "limited offer now", "")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("classify status: got %d want %d", resp.StatusCode, http.StatusOK)
	}
	classification := decodeJSON[classifyResp](t, resp)
	if classification.Category != "spam" {
		t.Fatalf("classify category: got %q want %q", classification.Category, "spam")
	}

	var scores map[string]float64
	resp = sendRequest(t, server, http.MethodPost, "/score", "limited offer now", "")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("score status: got %d want %d", resp.StatusCode, http.StatusOK)
	}
	scores = decodeJSON[map[string]float64](t, resp)
	if scores["spam"] <= scores["ham"] {
		t.Fatalf("expected spam score > ham score, got spam=%f ham=%f", scores["spam"], scores["ham"])
	}

	type infoResp struct {
		Categories map[string]json.RawMessage
	}
	resp = sendRequest(t, server, http.MethodGet, "/info", "", "")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("info status: got %d want %d", resp.StatusCode, http.StatusOK)
	}
	info := decodeJSON[infoResp](t, resp)
	if _, ok := info.Categories["spam"]; !ok {
		t.Fatal("expected spam category in info response")
	}
	if _, ok := info.Categories["ham"]; !ok {
		t.Fatal("expected ham category in info response")
	}

	type flushResp struct {
		Success    bool
		Categories map[string]json.RawMessage
	}
	resp = sendRequest(t, server, http.MethodPost, "/flush", "", "")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("flush status: got %d want %d", resp.StatusCode, http.StatusOK)
	}
	flush := decodeJSON[flushResp](t, resp)
	if !flush.Success {
		t.Fatal("expected flush success=true")
	}
	if len(flush.Categories) != 0 {
		t.Fatalf("expected no categories after flush, got %d", len(flush.Categories))
	}
}

func TestIntegrationAuthAndProbes(t *testing.T) {
	server := startIntegrationServer(t, "secret-token")

	type errorResp struct {
		Error string
	}

	resp := sendRequest(t, server, http.MethodGet, "/info", "", "")
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("unauth info status: got %d want %d", resp.StatusCode, http.StatusUnauthorized)
	}
	if got := resp.Header.Get("WWW-Authenticate"); got != `Bearer realm="gobayes"` {
		t.Fatalf("unexpected WWW-Authenticate header: got %q", got)
	}
	errPayload := decodeJSON[errorResp](t, resp)
	if errPayload.Error == "" {
		t.Fatal("expected non-empty error field for unauthorized response")
	}

	resp = sendRequest(t, server, http.MethodGet, "/healthz", "", "")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("healthz status: got %d want %d", resp.StatusCode, http.StatusOK)
	}
	_ = decodeJSON[map[string]string](t, resp)

	resp = sendRequest(t, server, http.MethodGet, "/readyz", "", "")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("readyz status: got %d want %d", resp.StatusCode, http.StatusOK)
	}
	_ = decodeJSON[map[string]string](t, resp)

	resp = sendRequest(t, server, http.MethodGet, "/info", "", "secret-token")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("auth info status: got %d want %d", resp.StatusCode, http.StatusOK)
	}
	_ = decodeJSON[map[string]interface{}](t, resp)
}

func TestIntegrationErrorContracts(t *testing.T) {
	server := startIntegrationServer(t, "")

	type errorResp struct {
		Error string
	}

	resp := sendRequest(t, server, http.MethodPost, "/train/spam!", "text", "")
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("invalid category status: got %d want %d", resp.StatusCode, http.StatusNotFound)
	}
	badCategory := decodeJSON[errorResp](t, resp)
	if badCategory.Error == "" {
		t.Fatal("expected non-empty error for invalid category route")
	}

	resp = sendRequest(t, server, http.MethodGet, "/classify", "", "")
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("method not allowed status: got %d want %d", resp.StatusCode, http.StatusMethodNotAllowed)
	}
	if allow := resp.Header.Get("Allow"); allow != http.MethodPost {
		t.Fatalf("unexpected Allow header: got %q want %q", allow, http.MethodPost)
	}
	badMethod := decodeJSON[errorResp](t, resp)
	if badMethod.Error == "" {
		t.Fatal("expected non-empty error for method not allowed")
	}
}
