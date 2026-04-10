# Code taken from NeuralNine https://www.youtube.com/watch?v=a040VmmO-AY

import os # check if something exists
import json
import random


import nltk # for tokenization

import numpy as np # for working with arrays

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Architecture. (Feel free to tweak)
class ChatbotModel(nn.Module): # extends from nn.Module
    def __init__(self, input_size, output_size): # define constructor
        super(ChatbotModel, self).__init__()    # constructor is called like this

        self.fc1 = nn.Linear(input_size,128)   # fully connected layer 1. (Input size is the shape of X value)
                                                # Dimension of input_size (bag of words) and 128 neurons
                                                # input size is the input, which is then fed into the first hidden layer of 128 neurons

        self.fc2 = nn.Linear(128, 64)           # second hidden layer
        self.fc3 = nn.Linear(64,output_size)    # output_size is number of probabilities for each intent

        self.relu = nn.ReLU()                   # activation function
        
        self.dropout = nn.Dropout(0.5)          # dropout layer, 50% chance of something being included for regularization

    def forward(self, x):                       # forward propogation, x is input.
        x = self.relu(self.fc1(x))              # feed x through first layer (weights and biases), then apply ReLU (break linearity).
        x = self.dropout(x)                     # apply dropout
        x = self.relu(self.fc2(x))              # do it again for 2nd layer
        x = self.dropout(x)
        x = self.fc3(x)                         # don't apply ReLU for 3rd layer because we want soft max (output probabilities).

        return x

class ChatbotAssistant: # Uses the ChatbotModel
    def __init__(self, intents_path, function_mappings = None): # intents_path is JSON file
        self.model = None
        self.intents_path = intents_path

        # turn example sentences into numbers
        self.documents = []
        self.vocabulary = []
        # possible outputs
        self.intents = []
        self.intents_responses = {} # dictionary

        self.function_mappings = function_mappings

        self.X = None # a matrix
        self.y = None # the output vector

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()

        words = nltk.word_tokenize(text)                                # break sentences into words
        words = [lemmatizer.lemmatize(word.lower()) for word in words]  # break variations of words into their stem (synoymns)

        return words
    
    def bag_of_words(self, words):
        """ Turns a sentence into a vector that represents the membership of vocabulary words in a sentence
        
        """
        return [1 if word in words else 0 for word in self.vocabulary] # Check if input words is something already known in vocabulary.
    
    def parse_intents(self):
        lemmarizer = nltk.WordNetLemmatizer()

        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r') as f: # open JSON in reading mode as a dictionary
                intents_data = json.load(f)

            for intent in intents_data['intents']: # dictionary that contains: tags, patterns, and responses.
                if intent['tag'] not in self.intents:  # if we haven't seen this intent yet (prevent duplicates)
                    self.intents.append(intent['tag']) # add the tag
                    self.intents_responses[intent['tag']] = intent['responses'] # add the tag's responses

                for pattern in intent['patterns']: # model is trained off patterns
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)    # put new words in vocabulary
                    self.documents.append((pattern_words, intent['tag'])) # combo of words and intent. (X and y data in a tuple)

                self.vocabulary = sorted(set(self.vocabulary)) # sort list and  eliminate duplicates

    def prepare_data(self):
        bags = []
        indices = []

        for document in self.documents:
            words = document[0] # words = pattern_words
            bag = self.bag_of_words(words) # turns words into 1s and 0s

            intent_index = self.intents.index(document[1]) # intent_index = intent['tag']
                                                            # asks: what is the index of a word in the intents list

            bags.append(bag)
            indices.append(intent_index)
        
        self.X = np.array(bags) # bags of words (matrix)
        self.y = np.array(indices)  # vector of correct predictions

    def train_model(self, batch_size, lr, epochs):
        """ This method trains the model

        @param self
        @param batch_size is how many instances will be processed in parallel at once
        @param lr is how fast the model moves in the direction of steepest descent (learning rate)
        @epochs is how many times we see the same data
        """
        X_tensor = torch.tensor(self.X, dtype=torch.float32) # bag of words represntaion of all sentences
        y_tensor = torch.tensor(self.y, dtype=torch.long) # integer (correct classification)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = ChatbotModel(self.X.shape[1], len(self.intents))   # Dynamically gets dimensions of X (1 bag of words), determined by how many words is seen.
                                                                        # Output is probabilities for each intent

        criterion = nn.CrossEntropyLoss()   # loss function
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()               # random initialized model
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)  # outputs is what the model produces with random initialized parameters.
                                                    # compare model outputs to batch_y ("the actual truth") diff?
                loss.backward() # back propogate to get gradient (direction of improvement/ calculus chain rule)
                optimizer.step() # take a step in right direction based off lr (learning rate)
                running_loss += loss

            print(f"Epoch {epoch+1}: Loss: {running_loss / len(loader):.4f}") # how loss improves over time

    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)

        with open(dimensions_path, 'w') as f:
            json.dump({ 'input_size': self.X.shape[1], 'output_size': len(self.intents)}, f)

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as f:
            dimensions = json.load(f)

        self.model = ChatbotModel(dimensions['input_size'], dimensions['output_size']) # A model is needed to load weights
        self.model.load_state_dict(torch.load(model_path, weights_only=True))

    def process_message(self, input_message):
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)

        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        self.model.eval() # put model into evaluation mode
        with torch.no_grad(): # no grading (no longer training)
            predictions = self.model(bag_tensor)

        predicted_class_index = torch.argmax(predictions, dim=1).item() # pick index with highest activation
        predicted_intent = self.intents[predicted_class_index] # string that is picked

        if self.function_mappings: # run a function if theres one associated with an intent
            if predicted_intent in self.function_mappings:
                self.function_mappings[predicted_intent]()

        if self.intents_responses[predicted_intent]: # say a response
            return random.choice(self.intents_responses[predicted_intent]) # randomly pick 1 of the responses.
        else:
            None

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("%$:P!#")

if __name__ == '__main__':
    Caine = ChatbotAssistant('intents.json', function_mappings = {'censor': clear_terminal}) # set clear_terminal as a fucntion Caine can use.
    Caine.parse_intents()
    Caine.prepare_data()
    Caine.train_model(batch_size=8, lr=0.001, epochs=200)

    Caine.save_model('chatbot_mode.pth', 'dimensions.json')

    while True:
        message = input('Enter your message: ')
        if message == '/quit':
            break
        
        print(Caine.process_message(message))
    

"""
#Check that input is being tokenized and lemmatized
caine = ChatbotAssistant('intents.json')
print(caine.tokenize_and_lemmatize('run running runs ran'))
"""