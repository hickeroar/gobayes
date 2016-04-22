package main

import (
	"encoding/json"
	"flag"
	"io/ioutil"
	"net/http"

	"github.com/gorilla/mux"
	"github.com/hickeroar/gobayes/bayes"
)

// ClassifierAPI handles requests and holds our classifier
type ClassifierAPI struct {
	classifier bayes.Classifier
}

// RegisterRoutes sets up the routing for the API
func (c *ClassifierAPI) RegisterRoutes(r *mux.Router) {
	r.HandleFunc("/train/{category:[A-Za-z]+}", c.TrainHandler).Methods("POST")
}

// TrainHandler handles requests to train the classifier
func (c *ClassifierAPI) TrainHandler(w http.ResponseWriter, req *http.Request) {
	category := mux.Vars(req)["category"]
	body, err := ioutil.ReadAll(req.Body)
	if err != nil {
		panic("Unable to Read Request Body")
	}
	c.classifier.Train(category, string(body))
	jsonResponse, err := json.Marshal(NewStandardClassifierResponse(c, true))

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

	http.ListenAndServe(":"+*port, r)
}
