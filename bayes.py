import sys
import math
import os
import pickle
import re
import string

class Bayes_Classifier:

    def __init__(self, traingDir="training/"):
        '''This method initializes and trains the Naive Bayes Sentiment Classifier.  If a
        cache of a trained classifier has been stored, it loads this cache.  Otherwise,
        the system will proceed through training.  After running this method, the classifier
        is ready to classify input text.'''

        # __init__ function checks for pickle files,
        # Loads the dictionary with positive and negative words if pickle files exists
        # else, initialize them and train the classifier.
        self.trainingDir = traingDir
        if os.path.exists("negative_reviews.p") and os.path.exists("positive_reviews.p"):
            self.negative_reviews = self.load("negative_reviews.p")
            self.positive_reviews = self.load("positive_reviews.p")
            self.total_negative_docs = len(self.negative_reviews)  # total number of -ve documents
            self.total_positive_docs = len(self.positive_reviews)  # total number of +ve documents
            self.total_negative_features = sum(self.negative_reviews.values())  # total number of -ve words
            self.total_positive_features = sum(self.positive_reviews.values())  # total number of +ve words
        else:
            self.negative_reviews = {}
            self.positive_reviews = {}
            self.total_negative_docs = 0
            self.total_positive_docs = 0
            self.total_negative_features = 0
            self.total_positive_features = 0
            self.train()

    def train(self):
        '''Trains the Naive Bayes Sentiment Classifier.'''

        # Creates a list of all words in each file by getting the files
        # iterates through each word, adds it to the suitable dictionary and saves the pickle files.
        iFileList = []
        for fFileObj in os.walk(self.trainingDir):
            iFileList = fFileObj[2]
            break
        for filename in iFileList:
            try:
                review = filename.split('-')[1]
            except:
                print("Error in the file ",filename)
            if review == '1':
                self.total_negative_docs += 1
            elif review == '5':
                self.total_positive_docs += 1
            file_content = self.loadFile(self.trainingDir + filename)
            token_list = self.tokenize(file_content)
            for token in token_list:
                if review == '1' and token not in string.punctuation:
                    if token not in self.negative_reviews:
                        self.negative_reviews[token] = 1
                    else:
                        self.negative_reviews[token] += 1
                    self.total_negative_features += 1
                elif review == '5' and token not in string.punctuation:
                    if token not in self.positive_reviews:
                        self.positive_reviews[token] = 1
                    else:
                        self.positive_reviews[token] += 1
                    self.total_positive_features += 1
            self.save(self.negative_reviews, "negative_reviews.p")
            self.save(self.positive_reviews, "positive_reviews.p")

    def classify(self, sText):
        '''Given a target string sText, this function returns the most likely document
        class to which the target string belongs. This function should return one of three
        strings: "positive", "negative" or "neutral".
        '''

        # prior probabilities for positive and negative are calculated by considering the above count of variables
        # Calculates the conditional probabilities for positive and negative for each word in file. Obtains condidional
        # probabilities by multiplying. Compares them to predict the class.
        negative_prior_probability = float(self.total_negative_docs) / (self.total_negative_docs + self.total_positive_docs)
        positive_prior_probability = float(self.total_positive_docs) / (self.total_negative_docs + self.total_positive_docs)
        negative_conditional_probability = math.log(negative_prior_probability)
        positive_conditional_probability = math.log(positive_prior_probability)
        token_list = self.tokenize(sText)
        for token in token_list:
            if token not in string.punctuation:
                if token in self.negative_reviews:
                    feature_negative_conditional_probability = float(self.negative_reviews[token] + 1) / (
                                self.total_negative_features + (1 * (self.total_negative_docs + self.total_positive_docs)))
                else:
                    feature_negative_conditional_probability = float(1) / (self.total_negative_features + 1)
                negative_conditional_probability += math.log(feature_negative_conditional_probability)
                if token in self.positive_reviews:
                    feature_positive_conditional_probability = float(self.positive_reviews[token] + 1) / (
                                self.total_positive_features + (1 * (self.total_negative_docs + self.total_positive_docs)))
                else:
                    feature_positive_conditional_probability = float(1) / (self.total_positive_features + 1)
                positive_conditional_probability += math.log(feature_positive_conditional_probability)
        diff = negative_conditional_probability - positive_conditional_probability
        if diff > 1:
            return "negative"
        elif diff < -1:
            return "positive"
        else:
            return "neutral"



    def loadFile(self, sFilename):
        '''Given a file name, return the contents of the file as a string.'''

        f = open(sFilename, "r", errors='replace')
        sTxt = f.read()
        f.close()
        return sTxt

    def save(self, dObj, sFilename):
        '''Given an object and a file name, write the object to the file using pickle.'''

        f = open(sFilename, "wb")
        p = pickle.Pickler(f)
        p.dump(dObj)
        f.close()

    def load(self, sFilename):
        '''Given a file name, load and return the object stored in the file.'''

        f = open(sFilename, "rb")
        u = pickle.Unpickler(f)
        dObj = u.load()
        f.close()
        return dObj

    def tokenize(self, sText):
        '''Given a string of text sText, returns a list of the individual tokens that
        occur in that string (in order).'''

        lTokens = []
        sToken = ""
        for c in sText:
            if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\'" or c == "_" or c == '-':
                sToken += c
            else:
                if sToken != "":
                    lTokens.append(sToken)
                    sToken = ""
                if c.strip() != "":
                    lTokens.append(str(c.strip()))

        if sToken != "":
            lTokens.append(sToken)

        return lTokens

bc = Bayes_Classifier()
result = bc.classify("what a horrible movie")
print(result)
