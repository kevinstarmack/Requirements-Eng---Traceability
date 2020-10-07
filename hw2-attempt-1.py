# use natural language toolkit
import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
import numpy as np
import time
stemmer = LancasterStemmer()
# 3 classes of training data
training_data=[]
training_data.append({"class": "Operational", "sentence": "The estimator shall not apply recycled parts to the collision estimate if no available parts are returned"})
training_data.append({"class": "Operational", "sentence": "The adjuster shall request a recycled parts audit of the collision estimate"})
training_data.append({"class": "Operational", "sentence": "The ratings shall include categories for attempted use of recycled parts and actual use of recycled parts"})
training_data.append({"class": "Operational", "sentence": "The system shall be able to store new recycled parts"})
training_data.append({"class": "Operational", "sentence": "The system will update existing recycled parts"})
training_data.append({"class": "Operational", "sentence": "The system shall be able to delete recycled parts"})
training_data.append({"class": "Operational", "sentence": "The system shall record the transportation status of parts reserved"})
training_data.append({"class": "Operational", "sentence": "The estimator shall search for available recycled parts using damaged vehicle parts information"})
training_data.append({"class": "Operational", "sentence": "The estimator shall search for available recycled parts using a list of preferred parts suppliers"})
training_data.append({"class": "Operational", "sentence": "The system shall search for available recycled parts for the supplied vehicle parts and suppliers"})
training_data.append({"class": "Operational", "sentence": "The system shall retain the available recycled parts and suppliers returned from the search"})
training_data.append({"class": "Operational", "sentence": "The available recycled parts information and their suppliers shall be returned to the user"})
training_data.append({"class": "Operational", "sentence": "The estimator shall apply selected recycled parts to the collision estimate"})
training_data.append({"class": "Operational", "sentence": "The recycled parts search results provided to the estimator shall be retrieved by the system"})
training_data.append({"class": "Operational", "sentence": "The system shall display recycled parts in a vertical table by name"})
training_data.append({"class": "Operational", "sentence": "The system shall display recycled parts and the preferred repair facility"})
training_data.append({"class": "Operational", "sentence": "The audit report shall include the total number of recycled parts used in the estimate"})
training_data.append({"class": "Operational", "sentence": "The audit report shall include the number of available recycled parts from the original search results"})
training_data.append({"class": "Operational", "sentence": "The audit report shall include the percentage of available recycled parts used in the estimate"})
training_data.append({"class": "Usability", "sentence": "The user shall search for the preferred repair facility using vehicle location and radius in miles"})
training_data.append({"class": "Usability", "sentence": "The search radius shall be between 1 and 30 miles"})
training_data.append({"class": "Usability", "sentence": "The system shall locate the preferred repair facility with the highest ratings for the input criteria"})
training_data.append({"class": "Usability", "sentence": "The adjuster shall enter the preferred repair facility on the estimate assignment"})
training_data.append({"class": "Usability", "sentence": "The adjuster shall be able to override the preferred repair facility on the estimate assignment"})
training_data.append({"class": "Usability", "sentence": "The estimator shall search by zipcode with a radius of 30 miles if no parts are found for the preferred parts suppliers"})
training_data.append({"class": "Usability", "sentence": "The adjuster shall review the collision estimate"})
training_data.append({"class": "Usability", "sentence": "The audit report shall be displayed to the user"})
training_data.append({"class": "Usability", "sentence": "The user shall select to view the preferred repair facility ratings"})
training_data.append({"class": "Usability", "sentence": "The system shall resolve the zipcode for the street address city and state if zipcode is unknown"})
training_data.append({"class": "Usability", "sentence": "The current repair facility ratings shall be displayed to the user"})
training_data.append({"class": "Usability", "sentence": "The system shall have a customizable Look and Feel"})
training_data.append({"class": "Usability", "sentence": "The system shall have an intuitive user interface"})
training_data.append({"class": "Usability", "sentence": "The system shall have a interface that allows for the viewing of the graph and the data table"})
training_data.append({"class": "Usability", "sentence": "The system shall display the repair facilities in a vertical table by name"})
training_data.append({"class": "Usability", "sentence": "The system shall display the preferred repair facility in a graph"})
training_data.append({"class": "Usability", "sentence": "The system shall color code the repair facilities according to their distance variance from current location"})
training_data.append({"class": "Usability", "sentence": "The system shall allow modification of the display"})
training_data.append({"class": "Usability", "sentence": "The system shall filter data by ratings"})
training_data.append({"class": "Usability", "sentence": "The system shall provide a history report of changes made to the ratings"})
training_data.append({"class": "Usability", "sentence": "The system shall allow the user to save the preferred repair facility search results"})
training_data.append({"class": "Usability", "sentence": "The system shall display the recycled parts information and suppliers research results"})
training_data.append({"class": "Usability", "sentence": "The system shall allow the estimator to save the collision estimate"})
training_data.append({"class": "Usability", "sentence": "The system shall provide the search history to users"})
training_data.append({"class": "Usability", "sentence": "The system shall notify the adjuster when a estimator responds to a collision estimate request"})
training_data.append({"class": "Usability", "sentence": "The user shall be able to retrieve a map showing repair facility locations for a specified area"})
training_data.append({"class": "Usability", "sentence": "The estimator shall be notified of new estimate request after automatic synchronization with system"})
training_data.append({"class": "Usability", "sentence": "The estimator shall search for available recycled parts using damaged vehicle parts information"})
training_data.append({"class": "Usability", "sentence": "The estimator shall search for available recycled parts using a list of preferred parts suppliers"})
training_data.append({"class": "Usability", "sentence": "The system shall search for available recycled parts for the supplied vehicle parts and suppliers"})
training_data.append({"class": "Usability", "sentence": "The system shall retain the available recycled parts and suppliers returned from the search"})
training_data.append({"class": "Usability", "sentence": "The available recycled parts information and their suppliers shall be returned to the user"})
training_data.append({"class": "Usability", "sentence": "The estimator shall apply selected recycled parts to the collision estimate"})
training_data.append({"class": "Usability", "sentence": "The recycled parts search results provided to the estimator shall be retrieved by the system"})
training_data.append({"class": "Usability", "sentence": "The system shall display recycled parts in a vertical table by name"})
training_data.append({"class": "Usability", "sentence": "The system shall display recycled parts and the preferred repair facility"})
training_data.append({"class": "Usability", "sentence": "The system shall allow the adjuster to save the audit report"})
training_data.append({"class": "Security", "sentence": "The system shall allow the adjuster to save the audit report"})
training_data.append({"class": "Security", "sentence": "The audit report shall be available to other adjusters at later points in time"})
training_data.append({"class": "Security", "sentence": "The user shall enter new ratings based on the audit report"})
training_data.append({"class": "Security", "sentence": "The system will allow priviledged users to view repair facility visiting schedules in multiple reporting views"})
training_data.append({"class": "Security", "sentence": "The audit report shall include the total number of recycled parts used in the estimate"})
training_data.append({"class": "Security", "sentence": "The audit report shall include the number of available recycled parts from the original search results"})
training_data.append({"class": "Security", "sentence": "The audit report shall include the percentage of available recycled parts used in the estimate"})
training_data.append({"class": "None", "sentence": "The vehicle location shall include street address city state and zip-code"})
training_data.append({"class": "None", "sentence": "The preferred repair facility shall be returned to the user"})
training_data.append({"class": "None", "sentence": "The system shall return a list of repair facilities within the radius if the preferred repair facility cannot be determined"})
training_data.append({"class": "None", "sentence": "The audit report shall include the total score of the audit which sums the individual line items"})
training_data.append({"class": "None", "sentence": "The system will display a blank set of ratings if there are not ratings yet defined"})
training_data.append({"class": "None", "sentence": "The ratings shall be from a scale of 1-10"})
training_data.append({"class": "None", "sentence": "The user shall select to save the preferred repair facility ratings"})
training_data.append({"class": "None", "sentence": "The preferred repair facility ratings shall be saved by the system"})
training_data.append({"class": "None", "sentence": "The vehicle data shall include vehicle year make and model"})
training_data.append({"class": "None", "sentence": "The system shall record repair facility visiting schedule entries"})
training_data.append({"class": "None", "sentence": "The system will notify users of their repair facility visiting schedules"})
training_data.append({"class": "None", "sentence": "The system shall have the ability to send repair facility visiting schedule reminders to users"})
training_data.append({"class": "None", "sentence": "The system shall send the repair facility contact information to schedule creaters"})
training_data.append({"class": "None", "sentence": "The system will record repair facility visiting schedule acknowledgments"})
training_data.append({"class": "None", "sentence": "The system shall store new conference rooms"})
training_data.append({"class": "None", "sentence": "The system shall update existing conference rooms"})
training_data.append({"class": "None", "sentence": "The system will be able to delete conference rooms"})
training_data.append({"class": "None", "sentence": "Each time a conference room is reserved the conference room schedule shall be updated to reflect the time and date of the reservation"})
training_data.append({"class": "None", "sentence": "The system shall display a map of the repair facility building showing conference room locations"})
training_data.append({"class": "None", "sentence": "The system shall record updated repair facility visiting schedule agendas"})
training_data.append({"class": "None", "sentence": "The system shall send a repair facility visiting schedule confirmation to the schedule creaters"})
training_data.append({"class": "None", "sentence": "The system shall record different schedule types"})
training_data.append({"class": "None", "sentence": "The system shall record all the recycled parts that has been reserved"})
training_data.append({"class": "None", "sentence": "The system shall notify repair facility of recycled parts transport requests"})
training_data.append({"class": "None", "sentence": "The system shall be able to send repair facility visiting schedule notifications via different kinds of end-user specified methods"})
training_data.append({"class": "None", "sentence": "The system will display an available status for unreserved conference rooms"})
training_data.append({"class": "None", "sentence": "The display shall have two regions: left of the display is graphical right of the display is a data table"})
training_data.append({"class": "None", "sentence": "The data displayed in both the nodes within the graph and the rows in the table are Summary data"})
training_data.append({"class": "None", "sentence": "The system shall offer the ability to pause and resume the refresh of data"})
training_data.append({"class": "Operational", "sentence": "The system shall generate an audit report based on the available recycled parts and the collision estimate"})
training_data.append({"class": "Usability", "sentence": "The system shall generate an audit report based on the available recycled parts and the collision estimate"})
training_data.append({"class": "Security", "sentence": "The system shall generate an audit report based on the available recycled parts and the collision estimate"})

#print ("%s sentences in training data" % len(training_data))
words = []
classes = []
documents = []
ignore_words = ['?,, shall, to, the, a']
# loop through each sentence in our training data
for pattern in training_data:
    # tokenize each word in the sentence
    w = nltk.word_tokenize(pattern['sentence'])
    # add to our words list
    words.extend(w)
    # add to documents in our corpus
    documents.append((w, pattern['class']))
    # add to our classes list
    if pattern['class'] not in classes:
        classes.append(pattern['class'])

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = list(set(words))

# remove duplicates
classes = list(set(classes))

#print (len(documents), "documents")
#print (len(classes), "classes", classes)
#print (len(words), "unique stemmed words", words)

# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    training.append(bag)
    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    output.append(output_row)

# sample training/output
i = 0
w = documents[i][0]
#print ([stemmer.stem(word.lower()) for word in w])
#print (training[i])
print (output[i])

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

def think(sentence, show_details=False):
    x = bow(sentence.lower(), words, show_details)
    if show_details:
        print ("sentence:", sentence, "\n bow:", x)
    # input layer is our bag of words
    l0 = x
    # matrix multiplication of input and hidden layer
    l1 = sigmoid(np.dot(l0, synapse_0))
    # output layer
    l2 = sigmoid(np.dot(l1, synapse_1))
    return l2

def train(X, y, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):

    print ("Training with %s neurons, alpha:%s, dropout:%s %s" % (hidden_neurons, str(alpha), dropout, dropout_percent if dropout else '') )
    print ("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X),len(X[0]),1, len(classes)) )
    np.random.seed(1)

    last_mean_error = 1
    # randomly initialize our weights with mean 0
    synapse_0 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2*np.random.random((hidden_neurons, len(classes))) - 1

    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    prev_synapse_1_weight_update = np.zeros_like(synapse_1)

    synapse_0_direction_count = np.zeros_like(synapse_0)
    synapse_1_direction_count = np.zeros_like(synapse_1)

    for j in iter(range(epochs+1)):

        # Feed forward through layers 0, 1, and 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))

        if(dropout):
            layer_1 *= np.random.binomial([np.ones((len(X),hidden_neurons))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))

        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        # how much did we miss the target value?
        layer_2_error = y - layer_2

        if (j% 10000) == 0 and j > 5000:
            # if this 10k iteration's error is greater than the last iteration, break out
            if np.mean(np.abs(layer_2_error)) < last_mean_error:
                print ("delta after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))) )
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                print ("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error )
                break

        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        layer_1_error = layer_2_delta.dot(synapse_1.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))

        if(j > 0):
            synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0))

        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update

        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update

    now = datetime.datetime.now()

    # persist synapses
    synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'words': words,
               'classes': classes
              }
    synapse_file = "synapses.json"

    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)
    print ("saved synapses to:", synapse_file)

X = np.array(training)
y = np.array(output)

start_time = time.time()

train(X, y, hidden_neurons=20, alpha=0.1, epochs=100000, dropout=False, dropout_percent=0.2)

elapsed_time = time.time() - start_time
print ("processing time:", elapsed_time, "seconds")
# probability threshold
ERROR_THRESHOLD = 0.2
# load our calculated synapse values
synapse_file = 'synapses.json'
with open(synapse_file) as data_file:
    synapse = json.load(data_file)
    synapse_0 = np.asarray(synapse['synapse0'])
    synapse_1 = np.asarray(synapse['synapse1'])

def classify(sentence, show_details=False):
    results = think(sentence, show_details)

    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD ]
    results.sort(key=lambda x: x[1], reverse=True)
    return_results =[[classes[r[0]],r[1]] for r in results]
    print ("%s \n classification: %s" % (sentence, return_results))
    return return_results
