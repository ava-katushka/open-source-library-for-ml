__author__ = 'irina'
# -*- coding: utf-8 -*-

import re
import logging
import time
import datetime

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

from bs4 import BeautifulSoup

import numpy as np
import pandas as pd

from gensim.models import Word2Vec

import sys
sys.path.insert(0, '..')
import CNNTextClassifier

import nltk
# nltk.download()  # Download text data sets, including stop words

from nltk.corpus import stopwords # Import the stop word list

import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)


def review_to_words(raw_review):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text()
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return " ".join( meaningful_words )


def text_to_wordlist(text, remove_stopwords=False):
    text = re.sub("[^a-zA-Z]", " ", text)
    words = text.lower().split()
    # Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    # 5. Return a list of words
    return words


def make_feature_matrix(words, model):
    # feature_matrix = np.zeros((len(words), num_features), dtype="float32")
    feature_matrix = []
    # counter = 0.
    for word in words:
        if word in model.vocab:
            feature_matrix.append(list(model[word]))
            # feature_matrix[counter] = model[word]
            # counter += 1
        # else:
        #    print 'word', word, 'is not in a model\n'
    feature_matrix = np.array(feature_matrix)
    return feature_matrix


def review_to_wordlist( review, remove_stopwords=False ):
    """
    Function to convert a document to a sequence of words,
    optionally removing stop words.  Returns a list of words.
    """
    # Remove HTML
    review_text = BeautifulSoup(review).get_text()
    return text_to_wordlist(review_text, remove_stopwords)


# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences


def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter % 5000. == 0.:
           print("Review %d of %d" % (counter, len(reviews)))
       #
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs


def simple_load_and_test():
    print "Loading data..."
    test_data = pd.read_csv("./testData.tsv",
                            header=0, delimiter="\t", quoting=3)
    print "size of test data = %d" % test_data.shape[0]

    print "Translating reviews to raw text format..."
    x_test = []
    max_count = 500 # test_data.shape[0]
    print "max_count = %d" % max_count
    for review in test_data["review"][0:max_count]:
        review_text = BeautifulSoup(review).get_text()
        x_test.append(review_text)

    classifier = CNNTextClassifier.TextClassifier(model_path="../100features_40minwords_10context")
    print "Loading state for classifier..."
    classifier.load("cnn_state_last")

    print "Prediction..."
    result = classifier.predict(x_test)
    result = np.array(result)
    result = result.flatten(1)
    # Write the test results
    output = pd.DataFrame(data={"id": test_data["id"][0:max_count], "sentiment": result})
    output.to_csv("cnn_word2vec_test20.05.2015.csv", index=False, quoting=3)


def main():
    print "Loading data..."
    train = pd.read_csv("labeledTrainData.tsv",
                        header=0, delimiter="\t", quoting=3)
    test_data = pd.read_csv("testData.tsv",
                            header=0, delimiter="\t", quoting=3)

    print "size of train data = %d, size of test data = %d" % (train.shape[0],
                                                               test_data.shape[0])

    print "Translating reviews to raw text format..."
    x_train = []
    max_count = 1000  #train.shape[0]
    print "number of samples = %d" % max_count

    print "Translating reviews to raw text format..."
    for review in train["review"][0:max_count]:
        review_text = BeautifulSoup(review).get_text()
        x_train.append(review_text)

    x_test = []
    for review in test_data["review"][0:max_count]:
        review_text = BeautifulSoup(review).get_text()
        x_test.append(review_text)

    classifier = CNNTextClassifier.TextClassifier(learning_rate=0.1, output_type='softmax',
                                               seed=0,
                                               model_path="../100features_40minwords_10context")
    #classifier = TextClassifier.TextClassifier(word_dimension=400,
    #                                           model_path="../word2vec.model")

    '''
    classifier.ready()
    print "Loading state for classifier..."
    classifier.load("cnn_state_last")
    '''

    print "Fitting a cnn to labeled training data..."
    y_train = np.array(train["sentiment"][0:max_count], dtype='int32')
    x_train = np.array(x_train)

    try:
        classifier.fit(x_train, y_train, n_epochs=100)
    except:
        new_state_path = "cnn_state_" + datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        print "Saving state to '%s'..." % new_state_path
        classifier.save_state(new_state_path)
        raise

    new_state_path = "cnn_state_" + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    print "Saving state to '%s'..." % new_state_path
    classifier.save_state(new_state_path)

    '''
    print "Predicting test results..."
    # Test & extract results
    result = classifier.predict(x_test)
    # TODO: избавиться от необходимости это делать
    result = np.array(result)
    result = result.flatten(1)


    # Write the test results
    print "Save test results..."
    output = pd.DataFrame(data={"id": test_data["id"], "sentiment": result})
    path_for_results = "test_results_" + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    output.to_csv(path_for_results, index=False, quoting=3)
    '''

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

'''

unlabeled_train = pd.read_csv( "./moviReviews/unlabeledTrainData.tsv", header=0,
                               delimiter="\t", quoting=3 )

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

sentences = []  # Initialize an empty list of sentences
print("Parsing sentences from training set")
for i, review in enumerate(train["review"]):
    sentences += review_to_sentences(review.decode("utf8"), tokenizer)

print("Parsing sentences from unlabeled set")
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review.decode("utf8"), tokenizer)

print len(sentences)
print sentences[0]
print sentences[1]

# Set values for various parameters
num_features = 100    # Word vector dimensionality
min_word_count = 40   # Minimum word count
num_workers = 8       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
print("Training model...")
model = Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "100features_40minwords_10context"
model.save(model_name)
'''