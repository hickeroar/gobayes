package bayes

import (
	"strings"
	"unicode"

	"github.com/hickeroar/gobayes/v3/bayes/stopwords"
	"github.com/kljensen/snowball"
	"golang.org/x/text/cases"
	"golang.org/x/text/language"
	"golang.org/x/text/unicode/norm"
)

// snowballLang maps our language names to kljensen/snowball language identifiers.
var snowballLang = map[string]string{
	"english":   "english",
	"spanish":   "spanish",
	"french":    "french",
	"russian":   "russian",
	"swedish":   "swedish",
	"norwegian": "norwegian",
	"hungarian": "hungarian",
}

// languageTag maps our language names to golang.org/x/text/language tags.
var languageTag = map[string]language.Tag{
	"english":   language.English,
	"spanish":   language.Spanish,
	"french":    language.French,
	"russian":   language.Russian,
	"swedish":   language.Swedish,
	"norwegian": language.Norwegian,
	"hungarian": language.Hungarian,
}

// NewDefaultTokenizer returns a tokenizer that normalizes (NFKC), lowercases
// with locale-aware case folding, splits on non-alphanumeric runes, stems with
// the given language, and optionally filters stop words. Language must be one of
// the seven supported by kljensen/snowball (english, spanish, french, russian,
// swedish, norwegian, hungarian); defaults to "english" if unsupported.
//
// If removeStopWords is true, tokens that are stop words for the language are
// filtered out. By default (removeStopWords false), stop words are kept.
//
// When snowball.Stem fails or returns empty, the original token is kept.
func NewDefaultTokenizer(lang string, removeStopWords bool) func(string) []string {
	lang = strings.ToLower(strings.TrimSpace(lang))
	if lang == "" {
		lang = "english"
	}
	if _, ok := snowballLang[lang]; !ok {
		lang = "english"
	}
	stemLang := snowballLang[lang]
	tag := languageTag[lang]
	lower := cases.Lower(tag)
	var stopSet map[string]struct{}
	if removeStopWords {
		stopSet = stopwords.Get(lang)
	}

	return func(sample string) []string {
		sample = norm.NFKC.String(sample)
		sample = lower.String(sample)
		rawTokens := strings.FieldsFunc(sample, func(r rune) bool {
			return !unicode.IsLetter(r) && !unicode.IsDigit(r)
		})

		tokens := make([]string, 0, len(rawTokens))
		for _, token := range rawTokens {
			stemmed, err := snowball.Stem(token, stemLang, true)
			if err == nil && stemmed != "" {
				token = stemmed
			}
			if stopSet != nil {
				if _, ok := stopSet[token]; ok {
					continue
				}
			}
			if token != "" {
				tokens = append(tokens, token)
			}
		}
		return tokens
	}
}
