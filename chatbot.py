import os, logging, sys
if __name__ == "__main__":
    if len(sys.argv) <= 1 or sys.argv[1] != "train":
        logging.disable(logging.WARNING)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
# things we need for Tensorflow
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import pandas as pd
import random
import json

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
NEURAL_NETWORK_FILE = os.path.join(FILE_DIR, "chatbot_neural_ntwk.h5")
DATA_FILE = os.path.join(FILE_DIR, "intents.json")

with open(DATA_FILE) as f:
    intents = json.load(f)

class Response:
    def __init__(self, response_messages, context):
        self.response_messages = response_messages
        self.context = context
    
    @property
    def is_context_empty():
        return self.context in [None, ""]

words = []
classes = []
documents = []
responses = {
    "": Response([], [""])
}
ignore_words = ['?']

# loop through each sentence in our intents patterns
for intent in intents['intents']:
    response_info = responses.get(intent['tag'], Response([], ""))
    response_info = Response(response_info.response_messages + intent["responses"], \
        response_info.response_messages + intent["context"])
    responses[intent['tag']] = response_info
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        # add to our words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
for tag in responses:
    responses[tag].response_messages = list(set(responses[tag].response_messages))
    responses[tag].context = list(set(responses[tag].context))

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# sort classes
classes = sorted(list(set(classes)))

def setup_training_data():
    # create our training data
    training = []
    # create an empty array for our output
    output_empty = [0] * len(classes)

    # training set, bag of words for each sentence
    for doc in documents:
        # initialize our bag of words
        bag = []
        # list of tokenized words for the pattern
        pattern_words = doc[0]
        # stem each word - create base word, in attempt to represent related words
        pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
        # create our bag of words array with 1, if word match found in current pattern
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)
        
        # output is a '0' for each tag and '1' for current tag (for each pattern)
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        
        training.append([bag, output_row])

    # shuffle our features and turn into np.array
    random.shuffle(training)
    training = np.array(training)
    # create train and test lists. X - patterns, Y - intents
    train_x = list(training[:,0])
    train_y = list(training[:,1])
    return train_x, train_y

def create_chatbot_model(train_x, train_y):

    # Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
    # equal to number of intents to predict output intent with softmax
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
    return model

def train_save_model():
    train_x, train_y = setup_training_data()
    model = create_chatbot_model(train_x, train_y)
    model.save(NEURAL_NETWORK_FILE)
    del model
    tf.keras.backend.clear_session()

def load_model():
    return tf.keras.models.load_model(NEURAL_NETWORK_FILE)

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words
    # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def classify_local(sentence, verbose=False):
    ERROR_THRESHOLD = 0.25
    model = load_model()
    
    # generate probabilities from the model
    input_data = pd.DataFrame([bow(sentence, words, show_details=verbose)], dtype=float, index=['input'])
    results = model.predict([input_data])[0]
    # filter out predictions below a threshold, and provide intent index
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], str(r[1])))
    # return tuple of intent and probability
    del model
    tf.keras.backend.clear_session()
    
    return return_list

def respond(sentence, label=None):
    if not os.path.isfile(NEURAL_NETWORK_FILE):
        print("Begin training model")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
        train_save_model()
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
        print("finished training model\n")
    if label == None:
        label_prob_list = classify_local(sentence)
        label = max(label_prob_list, key = lambda x: x[1])[0]
    return random.choice(responses[label].response_messages), label, random.choice(responses[label].context)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train": 
        train_save_model()
        logging.disable(logging.WARNING)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    ctx_exists = True
    ctx = None
    while ctx_exists:
        s = input("say something:")
        response, _, ctx = respond(s, ctx)
        if ctx == "":
            ctx = None
        print("chatbot:", response)
