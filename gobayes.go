package main

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"flag"
	"io"
	"log"
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
		mux := http.NewServeMux()

		controller := new(ClassifierAPI)
		controller.classifier = bayes.NewClassifier()
		controller.ready.Store(true)
		controller.RegisterRoutes(mux)

		port := flag.String("port", "8000", "The port the server should listen on.")
		authToken := flag.String("auth-token", "", "Optional bearer token required for all endpoints except /healthz and /readyz.")
		flag.Parse()

		var handler http.Handler = mux
		if *authToken != "" {
			handler = withAuthorizationToken(handler, *authToken)
		}

		server := newServer(":"+*port, handler)
		log.Printf("Server is listening on port %s.", *port)

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

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]string{"error": message})
}

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

func requireMethod(w http.ResponseWriter, req *http.Request, method string) bool {
	if req.Method != method {
		w.Header().Set("Allow", method)
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return false
	}
	return true
}

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

func main() {
	if err := runMain(); err != nil {
		logFatal(err)
	}
}
