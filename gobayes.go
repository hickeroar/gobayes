package main

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"regexp"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/hickeroar/gobayes/v3/bayes"
)

const maxRequestBodyBytes = 1 << 20 // 1 MiB

var categoryPathPattern = regexp.MustCompile(`^[-_A-Za-z0-9]+$`)

// serverConfig holds server and classifier configuration loaded from env and flags.
type serverConfig struct {
	Host             string
	Port             string
	AuthToken        string
	Language         string
	RemoveStopWords  bool
	Verbose          bool
}

// envOrDefault returns getenv(key) trimmed; if empty, returns def. Used for string env vars.
func envOrDefault(getenv func(string) string, key, def string) string {
	s := strings.TrimSpace(getenv(key))
	if s == "" {
		return def
	}
	return s
}

// envBool returns true if getenv(key) trimmed and lowercased is "1", "true", or "yes"; else false. Empty uses def.
func envBool(getenv func(string) string, key string, def bool) bool {
	val := strings.ToLower(strings.TrimSpace(getenv(key)))
	if val == "" {
		return def
	}
	switch val {
	case "1", "true", "yes":
		return true
	default:
		return false
	}
}

// setUsageDoubleDash sets fs.Usage so that PrintDefaults shows --flag instead of -flag.
func setUsageDoubleDash(fs *flag.FlagSet) {
	fs.Usage = func() {
		fmt.Fprintf(fs.Output(), "Usage of %s:\n", fs.Name())
		fs.VisitAll(func(f *flag.Flag) {
			fmt.Fprintf(fs.Output(), "  --%s\n    \t%s\n", f.Name, f.Usage)
		})
	}
}

// loadServerConfig loads server config from getenv (for defaults) and fs/args (CLI overrides). Parses flags.
func loadServerConfig(fs *flag.FlagSet, args []string, getenv func(string) string) (*serverConfig, error) {
	setUsageDoubleDash(fs)
	hostDefault := envOrDefault(getenv, "GOBAYES_HOST", "0.0.0.0")
	portDefault := envOrDefault(getenv, "GOBAYES_PORT", "8000")
	authDefault := envOrDefault(getenv, "GOBAYES_AUTH_TOKEN", "")
	langDefault := envOrDefault(getenv, "GOBAYES_LANGUAGE", "english")
	removeStopDefault := envBool(getenv, "GOBAYES_REMOVE_STOP_WORDS", false)
	verboseDefault := envBool(getenv, "GOBAYES_VERBOSE", false)

	hostFlag := fs.String("host", hostDefault, "Host interface to bind. (default: 0.0.0.0)")
	portFlag := fs.String("port", portDefault, "Port to bind. (default: 8000)")
	authFlag := fs.String("auth-token", authDefault, "Optional bearer token for non-probe endpoints.")
	languageFlag := fs.String("language", langDefault, "Language code for stemmer and stop words. (default: english)")
	removeStopFlag := fs.Bool("remove-stop-words", removeStopDefault, "Filter common stop words (the, is, and, etc.).")
	verboseFlag := fs.Bool("verbose", verboseDefault, "Log requests, responses, and classifier operations to stderr.")

	if err := fs.Parse(args); err != nil {
		return nil, err
	}

	language := strings.ToLower(strings.TrimSpace(*languageFlag))
	if language == "" {
		language = "english"
	}
	host := strings.TrimSpace(*hostFlag)
	if host == "" {
		host = "0.0.0.0"
	}
	port := strings.TrimSpace(*portFlag)
	if port == "" {
		port = "8000"
	}

	return &serverConfig{
		Host:            host,
		Port:            port,
		AuthToken:       strings.TrimSpace(*authFlag),
		Language:        language,
		RemoveStopWords: *removeStopFlag,
		Verbose:         *verboseFlag,
	}, nil
}

type httpServer interface {
	ListenAndServe() error
	Shutdown(ctx context.Context) error
}

var (
	makeSignalChannel = func() chan os.Signal { return make(chan os.Signal, 1) }
	notifySignals     = func(c chan<- os.Signal, sig ...os.Signal) { signal.Notify(c, sig...) }
	newServer         = func(addr string, handler http.Handler) httpServer {
		return &http.Server{
			Addr:              addr,
			Handler:           handler,
			ReadHeaderTimeout: 5 * time.Second,
			ReadTimeout:       10 * time.Second,
			WriteTimeout:      10 * time.Second,
			IdleTimeout:       30 * time.Second,
		}
	}
	logFatal = func(v ...interface{}) { log.Fatal(v...) }
	runMain  = func() error {
		cfg, err := loadServerConfig(flag.CommandLine, os.Args[1:], os.Getenv)
		if err != nil {
			return err
		}

		mux := http.NewServeMux()
		controller := new(ClassifierAPI)
		controller.classifier = bayes.NewClassifierWithOptions(cfg.Language, cfg.RemoveStopWords)
		controller.ready.Store(true)
		controller.RegisterRoutes(mux)

		var handler http.Handler = mux
		if cfg.AuthToken != "" {
			handler = withAuthorizationToken(handler, cfg.AuthToken)
		}
		if cfg.Verbose {
			handler = withVerbose(handler)
		}

		addr := net.JoinHostPort(cfg.Host, cfg.Port)
		server := newServer(addr, handler)
		log.Printf("Server is listening on %s.", addr)

		go func() {
			if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
				logFatal(err)
			}
		}()

		sigCh := makeSignalChannel()
		notifySignals(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		controller.ready.Store(false)

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		return server.Shutdown(ctx)
	}
)

// ClassifierAPI serves classifier HTTP endpoints and shared classifier state.
type ClassifierAPI struct {
	classifier *bayes.Classifier
	ready      atomic.Bool
}

// RegisterRoutes registers all API routes on the provided ServeMux.
func (c *ClassifierAPI) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/info", c.InfoHandler)
	mux.HandleFunc("/train/", c.TrainHandler)
	mux.HandleFunc("/untrain/", c.UntrainHandler)
	mux.HandleFunc("/classify", c.ClassifyHandler)
	mux.HandleFunc("/score", c.ScoreHandler)
	mux.HandleFunc("/flush", c.FlushHandler)
	mux.HandleFunc("/healthz", HealthHandler)
	mux.HandleFunc("/readyz", c.ReadyHandler)
}

// writeJSON marshals a value and writes it as a JSON HTTP response.
func writeJSON(w http.ResponseWriter, status int, value interface{}) {
	jsonResponse, err := json.Marshal(value)
	if err != nil {
		http.Error(w, `{"error":"failed to marshal response"}`, http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if _, err := w.Write(jsonResponse); err != nil {
		log.Printf("failed to write response: %v", err)
	}
}

// writeError writes a JSON error payload with the given status code.
func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]string{"error": message})
}

// readBody reads a bounded request body and returns the payload string.
func readBody(w http.ResponseWriter, req *http.Request) (string, bool) {
	req.Body = http.MaxBytesReader(w, req.Body, maxRequestBodyBytes)
	defer req.Body.Close()

	body, err := io.ReadAll(req.Body)
	if err != nil {
		var maxBytesError *http.MaxBytesError
		if errors.As(err, &maxBytesError) {
			writeError(w, http.StatusRequestEntityTooLarge, "request body too large")
			return "", false
		}
		writeError(w, http.StatusBadRequest, "unable to read request body")
		return "", false
	}

	return string(body), true
}

// categoryFromPath extracts and validates a category from a route path.
func categoryFromPath(path, prefix string) (string, bool) {
	category := strings.TrimPrefix(path, prefix)
	if category == "" || strings.Contains(category, "/") {
		return "", false
	}

	if !categoryPathPattern.MatchString(category) {
		return "", false
	}

	return category, true
}

// requireMethod enforces a single HTTP method and writes 405 on mismatch.
func requireMethod(w http.ResponseWriter, req *http.Request, method string) bool {
	if req.Method != method {
		w.Header().Set("Allow", method)
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return false
	}
	return true
}

// withVerbose wraps a handler to log request and response to stderr when verbose is enabled.
func withVerbose(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		log.Printf("[gobayes] %s %s", req.Method, req.URL.Path)
		body, _ := io.ReadAll(req.Body)
		req.Body = io.NopCloser(strings.NewReader(string(body)))
		if len(body) > 0 {
			preview := string(body)
			if len(preview) > 200 {
				preview = preview[:200] + "..."
			}
			log.Printf("[gobayes] request body (%d bytes): %s", len(body), preview)
		}
		rec := &responseRecorder{ResponseWriter: w, status: http.StatusOK, body: &bytesBuffer{}}
		next.ServeHTTP(rec, req)
		log.Printf("[gobayes] response %d", rec.status)
		if rec.body.Len() > 0 {
			preview := rec.body.String()
			if len(preview) > 200 {
				preview = preview[:200] + "..."
			}
			log.Printf("[gobayes] response body (%d bytes): %s", rec.body.Len(), preview)
		}
	})
}

type responseRecorder struct {
	http.ResponseWriter
	status int
	body   *bytesBuffer
}

type bytesBuffer struct {
	b []byte
}

func (b *bytesBuffer) Write(p []byte) (int, error) {
	b.b = append(b.b, p...)
	return len(p), nil
}

func (b *bytesBuffer) Len() int   { return len(b.b) }
func (b *bytesBuffer) String() string { return string(b.b) }

func (r *responseRecorder) WriteHeader(code int) {
	r.status = code
	r.ResponseWriter.WriteHeader(code)
}

func (r *responseRecorder) Write(p []byte) (int, error) {
	r.body.Write(p)
	return r.ResponseWriter.Write(p)
}

// withAuthorizationToken wraps a handler with bearer-token authorization checks.
func withAuthorizationToken(next http.Handler, expectedToken string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if req.URL.Path == "/healthz" || req.URL.Path == "/readyz" {
			next.ServeHTTP(w, req)
			return
		}

		authHeader := req.Header.Get("Authorization")
		scheme, providedToken, ok := strings.Cut(authHeader, " ")
		if !ok || !strings.EqualFold(scheme, "Bearer") || providedToken == "" {
			w.Header().Set("WWW-Authenticate", `Bearer realm="gobayes"`)
			writeError(w, http.StatusUnauthorized, "unauthorized")
			return
		}

		if subtle.ConstantTimeCompare([]byte(providedToken), []byte(expectedToken)) != 1 {
			w.Header().Set("WWW-Authenticate", `Bearer realm="gobayes"`)
			writeError(w, http.StatusUnauthorized, "unauthorized")
			return
		}

		next.ServeHTTP(w, req)
	})
}

// InfoHandler returns the current classifier training state.
func (c *ClassifierAPI) InfoHandler(w http.ResponseWriter, req *http.Request) {
	if !requireMethod(w, req, http.MethodGet) {
		return
	}

	writeJSON(w, http.StatusOK, NewInfoClassifierResponse(c))
}

// TrainHandler trains a category using request body text.
func (c *ClassifierAPI) TrainHandler(w http.ResponseWriter, req *http.Request) {
	if !requireMethod(w, req, http.MethodPost) {
		return
	}

	category, ok := categoryFromPath(req.URL.Path, "/train/")
	if !ok {
		writeError(w, http.StatusNotFound, "invalid category route")
		return
	}

	body, ok := readBody(w, req)
	if !ok {
		return
	}

	_ = c.classifier.Train(category, body)
	writeJSON(w, http.StatusOK, NewTrainingClassifierResponse(c, true))
}

// UntrainHandler untrains a category using request body text.
func (c *ClassifierAPI) UntrainHandler(w http.ResponseWriter, req *http.Request) {
	if !requireMethod(w, req, http.MethodPost) {
		return
	}

	category, ok := categoryFromPath(req.URL.Path, "/untrain/")
	if !ok {
		writeError(w, http.StatusNotFound, "invalid category route")
		return
	}

	body, ok := readBody(w, req)
	if !ok {
		return
	}

	_ = c.classifier.Untrain(category, body)
	writeJSON(w, http.StatusOK, NewTrainingClassifierResponse(c, true))
}

// ClassifyHandler classifies request body text and returns the top match.
func (c *ClassifierAPI) ClassifyHandler(w http.ResponseWriter, req *http.Request) {
	if !requireMethod(w, req, http.MethodPost) {
		return
	}

	body, ok := readBody(w, req)
	if !ok {
		return
	}

	writeJSON(w, http.StatusOK, c.classifier.Classify(body))
}

// ScoreHandler returns per-category scores for request body text.
func (c *ClassifierAPI) ScoreHandler(w http.ResponseWriter, req *http.Request) {
	if !requireMethod(w, req, http.MethodPost) {
		return
	}

	body, ok := readBody(w, req)
	if !ok {
		return
	}

	writeJSON(w, http.StatusOK, c.classifier.Score(body))
}

// FlushHandler deletes all training data and gives us a fresh slate.
func (c *ClassifierAPI) FlushHandler(w http.ResponseWriter, req *http.Request) {
	if !requireMethod(w, req, http.MethodPost) {
		return
	}

	c.classifier.Flush()
	writeJSON(w, http.StatusOK, NewTrainingClassifierResponse(c, true))
}

// HealthHandler returns liveness status for process health checks.
func HealthHandler(w http.ResponseWriter, req *http.Request) {
	if !requireMethod(w, req, http.MethodGet) {
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// ReadyHandler returns readiness status for traffic checks.
func (c *ClassifierAPI) ReadyHandler(w http.ResponseWriter, req *http.Request) {
	if !requireMethod(w, req, http.MethodGet) {
		return
	}
	if !c.ready.Load() {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"status": "not ready"})
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ready"})
}

// main starts the Gobayes server process.
func main() {
	if err := runMain(); err != nil {
		logFatal(err)
	}
}
