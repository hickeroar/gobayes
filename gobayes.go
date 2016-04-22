package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"net/http"

	"./bayes"
	"github.com/gorilla/mux"
)

// ClassifierAPI handles requests and holds our classifier instance
type ClassifierAPI struct {
	classifier bayes.Classifier
}

// RegisterRoutes sets up the routing for the API
func (c *ClassifierAPI) RegisterRoutes(r *mux.Router) {
	r.HandleFunc("/info", c.InfoHandler).Methods("GET")
	r.HandleFunc("/train/{category:[A-Za-z]+}", c.TrainHandler).Methods("POST")
	r.HandleFunc("/classify", c.ClassifyHandler).Methods("POST")
	r.HandleFunc("/score", c.ScoreHandler).Methods("POST")
	r.HandleFunc("/flush", c.FlushHander).Methods("POST")
}

// InfoHandler outputs the current state of training
func (c *ClassifierAPI) InfoHandler(w http.ResponseWriter, req *http.Request) {
	jsonResponse, _ := json.Marshal(NewInfoClassifierResponse(c))

	w.Header().Set("Content-Type", "application/json")
	w.Write(jsonResponse)
}

// TrainHandler handles requests to train the classifier
func (c *ClassifierAPI) TrainHandler(w http.ResponseWriter, req *http.Request) {
	category := mux.Vars(req)["category"]
	body, err := ioutil.ReadAll(req.Body)
	if err != nil {
		panic("Unable to Read Request Body")
	}
	if len(body) > 0 && len(category) > 0 {
		c.classifier.Train(category, string(body))
	}
	jsonResponse, _ := json.Marshal(NewTrainingClassifierResponse(c, true))

	w.Header().Set("Content-Type", "application/json")
	w.Write(jsonResponse)
}

// ClassifyHandler handles requests to classify samples of text
func (c *ClassifierAPI) ClassifyHandler(w http.ResponseWriter, req *http.Request) {
	body, err := ioutil.ReadAll(req.Body)
	if err != nil {
		panic("Unable to Read Request Body")
	}
	result := c.classifier.Classify(string(body))
	jsonResponse, _ := json.Marshal(result)

	w.Header().Set("Content-Type", "application/json")
	w.Write(jsonResponse)
}

// ScoreHandler handles returns the raw score data for a sample of text
func (c *ClassifierAPI) ScoreHandler(w http.ResponseWriter, req *http.Request) {
	body, err := ioutil.ReadAll(req.Body)
	if err != nil {
		panic("Unable to Read Request Body")
	}
	result := c.classifier.Score(string(body))
	jsonResponse, _ := json.Marshal(result)

	w.Header().Set("Content-Type", "application/json")
	w.Write(jsonResponse)
}

// FlushHander deletes all training data and gives us a fresh slate.
func (c *ClassifierAPI) FlushHander(w http.ResponseWriter, req *http.Request) {
	c.classifier.Flush()
	jsonResponse, _ := json.Marshal(NewTrainingClassifierResponse(c, true))

	w.Header().Set("Content-Type", "application/json")
	w.Write(jsonResponse)
}

func main() {
	r := mux.NewRouter()

	controller := new(ClassifierAPI)
	controller.classifier = *bayes.NewClassifier()
	controller.RegisterRoutes(r)

	port := flag.String("port", "8000", "The port the server should listen on.")
	flag.Parse()

	fmt.Println("Server is listening on port " + *port + ".")

	http.ListenAndServe(":"+*port, r)
}
