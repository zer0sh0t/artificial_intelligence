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

load_ds = True
save_ds = False
load = True
save = False

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

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

print(f'Dataset Status : Load:{load_ds} | Save:{save_ds}')
print(f'Model Status : Load:{load} | Save:{save}')

if load_ds:
    with open('datasets_r2/traning_set.pkl', 'rb') as f:
        training_set = pickle.load(f)
    with open('datasets_r2/testing_set.pkl', 'rb') as f:
        testing_set = pickle.load(f)
    with open('datasets_r2/word_features.pkl', 'rb') as f:
        word_features = pickle.load(f)
else:
    short_pos = open('movie_reviews/positive.txt', 'r').read()
    short_neg = open('movie_reviews/negative.txt', 'r').read()

    documents = []

    for l in short_pos.split('\n'):
        documents.append((l,'pos'))
    for l in short_neg.split('\n'):
        documents.append((l,'neg'))

    all_words = []

    short_pos_words = word_tokenize(short_pos)
    short_neg_words = word_tokenize(short_neg)

    for w in short_pos_words:
        all_words.append(w.lower())
    for w in short_pos_words:
        all_words.append(w.lower())

    all_words = nltk.FreqDist(all_words)
    word_features = list(all_words.keys())[:5000]

    featuresets = [(find_features(rev), category) for (rev, category) in documents]

    random.shuffle(featuresets)

    training_set = featuresets[:10000]
    testing_set = featuresets[10000:]

    if save_ds:
        with open('traning_set.pkl', 'wb') as f:
            pickle.dump(training_set, f)
        with open('testing_set.pkl', 'wb') as f:
            pickle.dump(testing_set, f)
        with open('word_features.pkl', 'wb') as f:
            pickle.dump(word_features, f)

if load:
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
else:
    classifier = nltk.NaiveBayesClassifier.train(training_set)

    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set)

    BNB_classifier = SklearnClassifier(BernoulliNB())
    BNB_classifier.train(training_set)

    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(training_set)

    NuSVC_classifier = SklearnClassifier(NuSVC())
    NuSVC_classifier.train(training_set)

    voted_classifier = VoteClasifier(classifier, MNB_classifier, BNB_classifier, LogisticRegression_classifier, NuSVC_classifier)

    if save:
        with open('naive_model.pkl', 'wb') as f:
            pickle.dump(classifier, f)
        with open('mn_model.pkl', 'wb') as f:
            pickle.dump(MNB_classifier, f)
        with open('bn_model.pkl', 'wb') as f:
            pickle.dump(BNB_classifier, f)
        with open('logistic_reg_model.pkl', 'wb') as f:
            pickle.dump(LogisticRegression_classifier, f)
        with open('nu_svc_model.pkl', 'wb') as f:
            pickle.dump(NuSVC_classifier, f)
        with open('voted_class_model.pkl', 'wb') as f:
            pickle.dump(voted_classifier, f)

print('naive accuracy percent:',nltk.classify.accuracy(classifier, testing_set) * 100)
print("BernoulliNB accuracy percent:",nltk.classify.accuracy(BNB_classifier, testing_set) * 100)
print("MultinomialNB accuracy percent:",nltk.classify.accuracy(MNB_classifier, testing_set) * 100)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

print('Classification:', voted_classifier.classify(testing_set[0][0]), 'Confidence:', voted_classifier.confidence(testing_set[0][0]) * 100)
print('Classification:', voted_classifier.classify(testing_set[1][0]), 'Confidence:', voted_classifier.confidence(testing_set[1][0]) * 100)
print('Classification:', voted_classifier.classify(testing_set[2][0]), 'Confidence:', voted_classifier.confidence(testing_set[2][0]) * 100)

# while True:
#     text = input('Type Something : (type quit to exit) ')
#     if text == 'quit':
#         break
#     text = find_features(text)
#
#     classification = voted_classifier.classify(text)
#     confidence = voted_classifier.confidence(text) * 100
#
#     if classification == 'neg':
#         if confidence <= 60:
#             classification = 'pos'
#             confidence = 90-confidence
#
#     print('Classification:', classification, 'Confidence:', confidence)
