# GoBayes
A memory-based, optional-persistence naive Bayesian text classification package and web API for Go.

---

## Why?
```
Bayesian text classification is often used for things like
spam detection, sentiment determination, or general categorization.

Essentially you collect samples of text that you know are of a certain
"type" or "category," then you use it to train a bayesian classifier.
Once you have trained the classifier with many samples of various
categories, you can begin to classify and/or score text samples to see
which category they fit best in.

You could, for instance, set up classification of sentiment by finding
samples of text that are happy, sad, angry, sarcastic, and so on, then
train a classifier using those samples. Once your classifier is trained,
you can begin to classify other text into one of those categories.

What a classifier does is look at text and tell you how much that text
"looks like" other categories of text that it has been trained for.
```


## Installation
```
$ git clone https://github.com/hickeroar/gobayes.git
$ cd gobayes
$ go build ./...
```

If you only want to use Gobayes as a library in your own app, add it as a dependency:
```
$ go get github.com/hickeroar/gobayes/v2
```

---

## Run as an API Server
```
$ go run .
Server is listening on port 8000.
```
```
go run . --help
Usage of gobayes:
  -auth-token string
        Optional bearer token required for all endpoints except /healthz and /readyz.
  -port string
        The port the server should listen on. (default "8000")
```
```
$ go run . --port 8181
Server is listening on port 8181.
```
```
$ go run . --port 8181 --auth-token my-secret-token
Server is listening on port 8181.
```

When `--auth-token` is set, all API endpoints except `/healthz` and `/readyz` require:

```
Authorization: Bearer <token>
```

Note: both single-hyphen and double-hyphen forms are accepted for flags, but examples use double-hyphen for consistency with common CLI conventions.

## Use as a Library in Your App

Import the library package:
```go
import "github.com/hickeroar/gobayes/v2/bayes"
```

Library example (train, classify, score, untrain, persist, restore):
```go
package main

import (
	"fmt"
	"log"

	"github.com/hickeroar/gobayes/v2/bayes"
)

func main() {
	// Create a new in-memory classifier instance.
	classifier := bayes.NewClassifier()

	// Train categories with representative text samples.
	classifier.Train("spam", "buy now limited offer click here")
	classifier.Train("ham", "team meeting schedule for tomorrow")

	// Classify a sample and inspect the top category + score.
	classification := classifier.Classify("limited offer today")
	fmt.Printf("category=%s score=%f\n", classification.Category, classification.Score)

	// Get per-category relative scores for another sample.
	scores := classifier.Score("team schedule update")
	fmt.Printf("scores=%v\n", scores)

	// Optionally remove previously trained text.
	classifier.Untrain("spam", "buy now limited offer click here")

	// Persist trained model state to a gob file.
	// Passing an empty path uses the default: /tmp/gobayes.gob.
	if err := classifier.SaveToFile("/tmp/model.gob"); err != nil {
		log.Fatalf("save failed: %v", err)
	}

	// Load persisted state into a fresh classifier instance.
	// Passing an empty path uses the default: /tmp/gobayes.gob.
	loaded := bayes.NewClassifier()
	if err := loaded.LoadFromFile("/tmp/model.gob"); err != nil {
		log.Fatalf("load failed: %v", err)
	}

	// Reuse the loaded model for classification.
	_ = loaded.Classify("limited offer today")
}
```

Notes for library usage:
- `Classifier` methods are goroutine-safe.
- Gobayes is memory-based with optional persistence when used as a library (`Save`/`Load`, `SaveToFile`/`LoadFromFile`).
- Persisted model data includes category/token tallies only; tokenizer configuration is runtime behavior and is not persisted.
- Scores are relative values and should be compared within the same model, not treated as calibrated probabilities.
- Category names accepted by `Train`/`Untrain` match `^[-_A-Za-z0-9]+$`; invalid names are ignored.
- Direct mutation of exported internals (for example `Classifier.Categories`) can bypass method-level synchronization.

For non-file workflows, you can use stream APIs:
- `Save(io.Writer) error`
- `Load(io.Reader) error`

File helper note:
- `SaveToFile` and `LoadFromFile` use `/tmp/gobayes.gob` when path is empty.
- When a path is provided, it must be absolute.

## Development Checks
```
$ go test ./...
$ go test -race ./...
```

---

## Using the HTTP API

### API Notes
- Category names in `/train/<category>` and `/untrain/<category>` must match `^[-_A-Za-z0-9]+$`.
- Request body size is capped at 1 MiB.
- Error responses use JSON format: `{"error":"<message>"}`.
- This service stores classifier state in memory only; restarting the process clears training data.

### Common Error Responses
| Status | When |
| --- | --- |
| `400` | Invalid request body |
| `404` | Invalid category route |
| `405` | Wrong HTTP method (`Allow` header is included) |
| `413` | Request body exceeds 1 MiB |

### Training the Classifier

##### Endpoint:
```
/train/<string:category>
Example: /train/spam
Accepts: POST
```
The result is of content-type "application/json" and contains a breakdown of each trained
category including the total text tokens that category contains, and the probabilities of
any given token existing in that category vs other categories.
```
{
    "Success": true,
    "Categories": {
        "ham": {
            "TokenTally": 1864,
            "ProbNotInCat": 0.540093757710338,
            "ProbInCat": 0.459906242289662
        },
        "spam": {
            "TokenTally": 2189,
            "ProbNotInCat": 0.4599062422896619,
            "ProbInCat": 0.5400937577103381
        }
    }
}
```
- The POST payload should contain the raw text that will train the classifier.
- You can train a category as many times as you want.


### Untraining the Classifier

##### Endpoint:
```
/untrain/<string:category>
Example: /untrain/spam
Accepts: POST
```
The result is of content-type "application/json" and contains a breakdown of each trained
category including the total text tokens that category contains, and the probabilities of
any given token existing in that category vs other categories.
```
{
    "Success": true,
    "Categories": {
        "ham": {
            "TokenTally": 1864,
            "ProbNotInCat": 0.540093757710338,
            "ProbInCat": 0.459906242289662
        },
        "spam": {
            "TokenTally": 2189,
            "ProbNotInCat": 0.4599062422896619,
            "ProbInCat": 0.5400937577103381
        }
    }
}
```
- The POST payload should contain the raw text that will untrain the classifier.
- If there are no remaining tokens in a category, that category will be removed.


### Getting Classifier Status

##### Endpoint:
```
/info
Accepts: GET
```
The result is of content-type "application/json" and contains a breakdown of each trained
category including the total text tokens that category contains, and the probabilities of
any given token existing in that category vs other categories.
```
{
    "Categories": {
        "ham": {
            "TokenTally": 1864,
            "ProbNotInCat": 0.540093757710338,
            "ProbInCat": 0.459906242289662
        },
        "spam": {
            "TokenTally": 2189,
            "ProbNotInCat": 0.4599062422896619,
            "ProbInCat": 0.5400937577103381
        }
    }
}
```
- No payload or parameters are expected.


### Classifying Text

##### Endpoint:
```
/classify
Accepts: POST
```
The result is of content-type "application/json" and contains a simple object
showing the Category classification and the classification Score that was calculated.
The Score should not be relied on as any kind of specific indicator, as it's a
relative score and varies based on the number of tokens your categories have been
trained with.
```
{
    "Category": "spam",
    "Score": 43.48754443957434
}
```
- The POST payload should contain the raw text that you want to classify.


### Scoring Text

##### Endpoint
```
/score
Accepts: POST
```
The result of content-type "application/json" and contains is a simple object showing
the scores achieved by each category. As with classification, the scores should not be
relied on as any kind of specific indicator, as it's a relative score and varies based
on the number of tokens your categories have been trained with.
```
{
    "ham": 5.512455560425657,
    "spam": 43.48754443957434
}
```
- The POST payload should contain the raw text that you want to score.


### Flushing Training Data

##### Endpoint
```
/flush
Accepts: POST
```
The result of content-type "application/json" and contains a the list of categories
that have been trained. This list should be empty, indicating that the classifier
has been flushed.
```
{
    "Success": true,
    "Categories": {}
}
```
- No payload or parameters are expected.

### Health and Readiness
##### Liveness endpoint
```
/healthz
Accepts: GET
```

##### Readiness endpoint
```
/readyz
Accepts: GET
```

`/healthz` and `/readyz` are intentionally unauthenticated so infrastructure probes can reach them even when API auth is enabled.

## Operational Notes
- The HTTP server stores training data in memory only. Process restarts and deploy rollouts wipe model state unless your app replays training events.
- When using Gobayes as a library, model state can be persisted and restored with `Save`/`Load` or `SaveToFile`/`LoadFromFile`.
- Treat Gobayes as stateful if you rely on trained categories. For production use, define how training data is restored after restart.
- `/readyz` returns `200` while accepting traffic and returns `503` when the process is draining during shutdown.
