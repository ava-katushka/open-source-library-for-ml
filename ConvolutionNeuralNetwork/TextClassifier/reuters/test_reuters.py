#!/usr/bin/python
# -*- coding: utf-8 -*-
import unittest
import xmlrunner
import sys
import numpy as np
sys.path.insert(0, '..')
sys.path.insert(0, './ConvolutionNeuralNetwork/TextClassifier')
sys.path.insert(0, './AML-TextClassification')
sys.path.insert(0, './ConvolutionNeuralNetwork/TextClassifier/reuters')
from CNNTextClassifier import CNNTextClassifier
import datetime
import time
from gensim.models import Word2Vec
#from nltk.corpus import stopwords
import re

sys.path.insert(0, '../../../AML-TextClassification')
import reuters
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

'''
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
'''

class TestReutersTextClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """ Загрузка данных, загрузка заранее обученного классификатора """
        print "Loading and parsing data..."
        data_path = './AML-TextClassification/data'
        rp = reuters.ReutersParser(data_path, multilabel=False)
        rp.parse()
        cls.X_train = rp.get_corpus("train", "topics", "data")
        cls.Y_train = rp.get_corpus("train", "topics", "target")

        cls.X_test = rp.get_corpus("test", "topics", "data")
        cls.Y_test = rp.get_corpus("test", "topics", "target")
        print "OK"
        assert len(cls.X_train) > 0
        assert len(cls.Y_train) > 0
        assert len(cls.X_test) > 0
        assert len(cls.Y_test) > 0
        n_out = max(cls.Y_train) + 1
        #model_name = "word2vec_100_reuters"
        #model = get_word2vec_model(X_train, X_test)
        #model.save(model_name)
        model_path = "../100features_40minwords_10context"

        cls.cnn_text_clf = CNNTextClassifier(learning_rate=0.1, window=5, n_hidden=10,
                                             n_filters=25,
                                             n_out=n_out, word_dimension=100, seed=1,
                                             model_path=model_path, L1_reg=0.1, L2_reg=0.1)

        new_state_path = 'cnn_best_100_state_2015-05-24-05:42:00'
        print "Loading state for classifier..."
        cls.cnn_text_clf.load(new_state_path)

    def test_dimensions(self):
        params = self.cnn_text_clf.get_cnn_params()
        self.assertEqual(len(params), 6)
        # Вектор смещения для первого свёрточого слоя
        self.assertEqual(len(params[5].get_value()), self.cnn_text_clf.n_filters)
        # Веса для первого свёрточого слоя
        self.assertEqual(params[4].get_value().shape, (self.cnn_text_clf.n_filters, 1,
                                                       self.cnn_text_clf.window,
                                                       self.cnn_text_clf.word_dimension))
        # Вектор смещения для полносвязного слоя
        self.assertEqual(params[3].get_value().shape, (self.cnn_text_clf.n_hidden,))
        # Веса для полносвязного слоя
        self.assertEqual(params[2].get_value().shape, (self.cnn_text_clf.n_filters,
                                                       self.cnn_text_clf.n_hidden))

    def test_score(self):
        print "Count score..."
        score = np.mean(self.cnn_text_clf.predict(self.X_test) == self.Y_test)
        print "result =", score
        precision = metrics.precision_score(self.Y_test, self.cnn_text_clf.predict(self.X_test))
        print "precision =", precision
        self.assertTrue(score > 0.8146)
        self.assertTrue(precision > 0.75)


if __name__ == '__main__':
    unittest.main(testRunner=xmlrunner.XMLTestRunner(output='test-reports'))

