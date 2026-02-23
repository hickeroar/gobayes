package stopwords

import (
	"sync"
	"strings"
)

var (
	mu    sync.RWMutex
	cache map[string]map[string]struct{}
)

func init() {
	cache = make(map[string]map[string]struct{})
	langs := map[string][]string{
		"english":   englishWords,
		"spanish":   spanishWords,
		"french":    frenchWords,
		"russian":   russianWords,
		"swedish":   swedishWords,
		"norwegian": norwegianWords,
		"hungarian": hungarianWords,
	}
	for lang, words := range langs {
		set := make(map[string]struct{}, len(words))
		for _, w := range words {
			set[strings.ToLower(w)] = struct{}{}
		}
		cache[lang] = set
	}
}

// Get returns the stopword set for the given language. The language must match
// one of SupportedLanguages (e.g. "english", "spanish"). Returns nil if the
// language is not supported.
func Get(lang string) map[string]struct{} {
	lang = strings.ToLower(strings.TrimSpace(lang))
	mu.RLock()
	defer mu.RUnlock()
	return cache[lang]
}

// Supported returns true if the language has a stopword list.
func Supported(lang string) bool {
	return Get(lang) != nil
}
