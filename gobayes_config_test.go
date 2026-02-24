package main

import (
	"bytes"
	"flag"
	"strings"
	"testing"
)

func TestEnvOrDefault_Empty(t *testing.T) {
	getenv := func(string) string { return "" }
	if got := envOrDefault(getenv, "K", "def"); got != "def" {
		t.Errorf("envOrDefault(empty) = %q, want def", got)
	}
}

func TestEnvOrDefault_WhitespaceOnly(t *testing.T) {
	getenv := func(string) string { return "  \t  " }
	if got := envOrDefault(getenv, "K", "def"); got != "def" {
		t.Errorf("envOrDefault(whitespace) = %q, want def", got)
	}
}

func TestEnvOrDefault_SetReturnsTrimmed(t *testing.T) {
	getenv := func(string) string { return "  val  " }
	if got := envOrDefault(getenv, "K", "def"); got != "val" {
		t.Errorf("envOrDefault(set) = %q, want val", got)
	}
}

func TestEnvBool_EmptyUsesDefault(t *testing.T) {
	getenv := func(string) string { return "" }
	if got := envBool(getenv, "K", true); got != true {
		t.Errorf("envBool(empty, true) = %v", got)
	}
	if got := envBool(getenv, "K", false); got != false {
		t.Errorf("envBool(empty, false) = %v", got)
	}
}

func TestEnvBool_Truthy(t *testing.T) {
	for _, val := range []string{"1", "true", "yes", "TRUE", "Yes"} {
		getenv := func(string) string { return val }
		if got := envBool(getenv, "K", false); got != true {
			t.Errorf("envBool(%q) = false, want true", val)
		}
	}
}

func TestEnvBool_Falsy(t *testing.T) {
	for _, val := range []string{"0", "false", "no", "x"} {
		getenv := func(string) string { return val }
		if got := envBool(getenv, "K", true); got != false {
			t.Errorf("envBool(%q, true) = true, want false", val)
		}
	}
}

func TestLoadServerConfig_Defaults(t *testing.T) {
	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	getenv := func(string) string { return "" }
	cfg, err := loadServerConfig(fs, []string{}, getenv)
	if err != nil {
		t.Fatalf("loadServerConfig: %v", err)
	}
	if cfg.Host != "0.0.0.0" || cfg.Port != "8000" || cfg.Language != "english" {
		t.Errorf("defaults: host=%q port=%q language=%q", cfg.Host, cfg.Port, cfg.Language)
	}
	if cfg.AuthToken != "" || cfg.RemoveStopWords || cfg.Verbose {
		t.Errorf("defaults: auth=%q removeStop=%v verbose=%v", cfg.AuthToken, cfg.RemoveStopWords, cfg.Verbose)
	}
}

func TestLoadServerConfig_EnvOverridesDefaults(t *testing.T) {
	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	getenv := func(key string) string {
		switch key {
		case "GOBAYES_HOST":
			return "127.0.0.1"
		case "GOBAYES_PORT":
			return "9000"
		case "GOBAYES_LANGUAGE":
			return "spanish"
		case "GOBAYES_REMOVE_STOP_WORDS":
			return "true"
		case "GOBAYES_VERBOSE":
			return "yes"
		default:
			return ""
		}
	}
	cfg, err := loadServerConfig(fs, []string{}, getenv)
	if err != nil {
		t.Fatalf("loadServerConfig: %v", err)
	}
	if cfg.Host != "127.0.0.1" || cfg.Port != "9000" || cfg.Language != "spanish" {
		t.Errorf("env: host=%q port=%q language=%q", cfg.Host, cfg.Port, cfg.Language)
	}
	if !cfg.RemoveStopWords || !cfg.Verbose {
		t.Errorf("env: removeStop=%v verbose=%v", cfg.RemoveStopWords, cfg.Verbose)
	}
}

func TestLoadServerConfig_FlagsOverrideEnv(t *testing.T) {
	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	getenv := func(key string) string {
		if key == "GOBAYES_PORT" {
			return "9000"
		}
		return ""
	}
	cfg, err := loadServerConfig(fs, []string{"--port", "8888", "--host", "0.0.0.0"}, getenv)
	if err != nil {
		t.Fatalf("loadServerConfig: %v", err)
	}
	if cfg.Port != "8888" {
		t.Errorf("flag should override env: port=%q", cfg.Port)
	}
}

func TestLoadServerConfig_SingleDashAccepted(t *testing.T) {
	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	getenv := func(string) string { return "" }
	cfg, err := loadServerConfig(fs, []string{"-port", "7777", "-verbose"}, getenv)
	if err != nil {
		t.Fatalf("loadServerConfig: %v", err)
	}
	if cfg.Port != "7777" || !cfg.Verbose {
		t.Errorf("single-dash: port=%q verbose=%v", cfg.Port, cfg.Verbose)
	}
}

func TestLoadServerConfig_LanguageNormalized(t *testing.T) {
	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	getenv := func(string) string { return "" }
	cfg, err := loadServerConfig(fs, []string{"--language", "  Spanish  "}, getenv)
	if err != nil {
		t.Fatalf("loadServerConfig: %v", err)
	}
	if cfg.Language != "spanish" {
		t.Errorf("language normalized: %q", cfg.Language)
	}
}

func TestLoadServerConfig_EmptyLanguageDefaultsToEnglish(t *testing.T) {
	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	getenv := func(string) string { return "" }
	cfg, err := loadServerConfig(fs, []string{"--language", ""}, getenv)
	if err != nil {
		t.Fatalf("loadServerConfig: %v", err)
	}
	if cfg.Language != "english" {
		t.Errorf("empty language: %q", cfg.Language)
	}
}

func TestLoadServerConfig_HelpShowsDoubleDash(t *testing.T) {
	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	var buf bytes.Buffer
	fs.SetOutput(&buf)
	getenv := func(string) string { return "" }
	_, _ = loadServerConfig(fs, []string{"--help"}, getenv)
	out := buf.String()
	for _, want := range []string{"--host", "--port", "--auth-token", "--language", "--verbose"} {
		if !strings.Contains(out, want) {
			t.Errorf("help output missing %q: %s", want, out)
		}
	}
}

func TestLoadServerConfig_UnknownFlagReturnsError(t *testing.T) {
	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	getenv := func(string) string { return "" }
	_, err := loadServerConfig(fs, []string{"--unknown", "x"}, getenv)
	if err == nil {
		t.Fatal("expected error for unknown flag")
	}
}

func TestLoadServerConfig_EmptyHostAndPortNormalized(t *testing.T) {
	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	getenv := func(string) string { return "" }
	// Pass empty string for host and port via flags (e.g. --host "" --port "")
	cfg, err := loadServerConfig(fs, []string{"--host", "", "--port", ""}, getenv)
	if err != nil {
		t.Fatalf("loadServerConfig: %v", err)
	}
	if cfg.Host != "0.0.0.0" || cfg.Port != "8000" {
		t.Errorf("empty host/port should normalize: host=%q port=%q", cfg.Host, cfg.Port)
	}
}
