# GoBayes
A web api providing memory-based naïve bayesian text classification.

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
$ go get github.com/hickeroar/gobayes
```

## Usage
```
$ gobayes
Server is listening on port 8000.
```
```
$ gobayes -help
Usage of gobayes:
  -port string
        The port the server should listen on. (default "8000")
```
```
$ gobayes -port 8181
Server is listening on port 8181.
```

## Training the Classifier

Endpoint:
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

## Getting current Classifier status information

Endpoint:
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

