import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import movie_reviews
from nltk.classify import ClassifierI
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from statistics import mode
import random
import pickle
from statistics import mode
from nltk.classify import ClassifierI
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC, NuSVC, LinearSVC

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

class VoteClasifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        most_common_count = votes.count(mode(votes))
        conf = most_common_count/len(votes)
        return conf

with open('datasets_r2/traning_set.pkl', 'rb') as f:
    training_set = pickle.load(f)
with open('datasets_r2/testing_set.pkl', 'rb') as f:
    testing_set = pickle.load(f)
with open('datasets_r2/word_features.pkl', 'rb') as f:
    word_features = pickle.load(f)

with open('models_r2/naive_model.pkl', 'rb') as f:
    classifier = pickle.load(f)
with open('models_r2/mn_model.pkl', 'rb') as f:
    MNB_classifier = pickle.load(f)
with open('models_r2/bn_model.pkl', 'rb') as f:
    BNB_classifier = pickle.load(f)
with open('models_r2/logistic_reg_model.pkl', 'rb') as f:
    LogisticRegression_classifier = pickle.load(f)
with open('models_r2/nu_svc_model.pkl', 'rb') as f:
    NuSVC_classifier = pickle.load(f)
with open('models_r2/voted_class_model.pkl', 'rb') as f:
    voted_classifier = pickle.load(f)

while True:
    text = input('Type Something : (type quit to exit) ')
    if text == 'quit':
        break
    text = find_features(text)

    classification = voted_classifier.classify(text)
    confidence = voted_classifier.confidence(text) * 100

    if classification == 'neg':
        if confidence <= 60:
            classification = 'pos'
            confidence = 90-confidence

    print('Classification:', classification, 'Confidence:', confidence)
