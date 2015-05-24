#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import numpy as np
sys.path.insert(0, '..')
from CNNTextClassifier import CNNTextClassifier
import datetime
import time
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import re

sys.path.insert(0, '../../../AML-TextClassification')
import reuters
from sklearn import metrics


def text_to_wordlist(text, remove_stopwords=False):
    text = re.sub("[^a-zA-Z]", " ", text)
    words = text.lower().split()
    # Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if (not w in stops) and (not w == 'reuter')]
    # 5. Return a list of words
    return words


def get_word2vec_model(X_train, X_test):
    sentences = []
    print("Parsing X_train...")
    for text in X_train:
        sentences.append(text_to_wordlist(text, True))

    print("Parsing X_test...")
    for text in X_test:
        sentences.append(text_to_wordlist(text, True))

    print "number of texts:", len(sentences)

    # Set values for various parameters
    num_features = 100    # Word vector dimensionality
    min_word_count = 10   # Minimum word count
    num_workers = 8       # Number of threads to run in parallel
    context = 5          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print("Training model...")
    model = None
    try:
        model = Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count,
                         window=context, sample=downsampling)
    except KeyboardInterrupt:
        print "Fit Interrupt, if you really want to interrupt execution, try again"
        model_name = "word2vec_100_reuters_not_finished"
        model.save(model_name)
        raise

    model.init_sims(replace=True)
    return model


def main():
    print "Loading and parsing data..."
    data_path = '../../../AML-TextClassification/data'
    rp = reuters.ReutersParser(data_path, multilabel=False)
    rp.parse()
    X_train = rp.get_corpus("train", "topics", "data")
    Y_train = rp.get_corpus("train", "topics", "target")

    X_test = rp.get_corpus("test", "topics", "data")
    Y_test = rp.get_corpus("test", "topics", "target")
    print "OK"

    n_out = max(Y_train) + 1
    #model_name = "word2vec_100_reuters"
    #model = get_word2vec_model(X_train, X_test)
    #model.save(model_name)
    model_path = "../100features_40minwords_10context"
    # learning_rate=0.1, window=5, n_hidden=20, n_filters=25,
    #                                 n_out=n_out, word_dimension=400
    cnn_text_clf = CNNTextClassifier(learning_rate=0.1, window=5, n_hidden=10, n_filters=25,
                                     n_out=n_out, word_dimension=100, seed=1,
                                     model_path=model_path, L1_reg=0.1, L2_reg=0.1)
    cnn_text_clf.ready()
    # print "test score before training:", cnn_text_clf.score(X_test, Y_test)

    new_state_path = 'cnn_best_100_state_2015-05-24-05:42:00'
    print "Loading state for classifier..."
    cnn_text_clf.load(new_state_path)
    print "Count score..."
    print "result =", np.mean(cnn_text_clf.predict(X_test) == Y_test)
    print "precision =", metrics.precision_score(Y_test, cnn_text_clf.predict(X_test))
'''
    try:
        cnn_text_clf.fit(X_train, Y_train, X_test[0:500], Y_test[0:500], n_epochs=50)
    except KeyboardInterrupt:
        print "Fit Interrupt, if you really want to interrupt execution, try again"

    new_state_path = "cnn_best_100_state_" + datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    print "Saving state to '%s'..." % new_state_path
    cnn_text_clf.save_state(new_state_path)

    print "TOTAL TEST SCORE:", cnn_text_clf.score(X_test, Y_test)
    metrics.precision_score(Y_test, cnn_text_clf.predict(X_test))
'''
if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
