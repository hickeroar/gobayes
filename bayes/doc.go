// Package bayes provides a Naive Bayes text classifier with JSON persistence
// and configurable tokenization.
//
// Tokenization: The default tokenizer (see NewDefaultTokenizer) normalizes
// text (NFKC), lowercases with locale-aware case folding, stems, and optionally
// filters stop words. When snowball.Stem fails or returns an empty string, the
// original token is kept.
package bayes
