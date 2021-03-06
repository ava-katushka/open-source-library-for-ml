# -*- coding: utf-8 -*-
import unittest
import xmlrunner
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

from CNNTextClassifier import CNNTextClassifier

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestTextClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """ Загрузка данных, загрузка заранее обученного классификатора """
        # -*- coding: utf-8 -*-
        print "Loading data..."
        data = pd.read_csv("./movieReviews/labeledTrainData.tsv",
                           header=0, delimiter="\t", quoting=3)
        print "size of data = %d" % data.shape[0]

        print "Translating reviews to raw text format..."
        x_train = []
        max_train_count = 500
        print "max_train_count = %d" % max_train_count
        for review in data["review"][0:max_train_count]:
            review_text = BeautifulSoup(review).get_text()
            x_train.append(review_text)

        cls.x_train = np.array(x_train)
        cls.y_train = np.array(data["sentiment"][0:max_train_count], dtype='int32')

        x_test = []
        max_test_count = 100
        print "max_test_count = %d" % max_test_count
        for review in data["review"][max_train_count:max_train_count + max_test_count]:
            review_text = BeautifulSoup(review).get_text()
            x_test.append(review_text)

        cls.x_test = np.array(x_test)
        cls.y_test = np.array(data["sentiment"][max_train_count:max_train_count + max_test_count],
                              dtype='int32')

        cls.classifier = CNNTextClassifier()
        cls.classifier.ready()

        #print "Loading state for classifier..."
        #cls.classifier.load("./movieReviews/cnn_state_20150523023906")

    def test_dimensions(self):
        params = self.classifier.get_cnn_params()
        self.assertEqual(len(params), 6)
        # Вектор смещения для первого свёрточого слоя
        self.assertEqual(len(params[5].get_value()), self.classifier.n_filters)
        # Веса для первого свёрточого слоя
        self.assertEqual(params[4].get_value().shape, (self.classifier.n_filters, 1,
                                                       self.classifier.window,
                                                       self.classifier.word_dimension))
        # Вектор смещения для полносвязного слоя
        self.assertEqual(params[3].get_value().shape, (self.classifier.n_hidden,))
        # Веса для полносвязного слоя
        self.assertEqual(params[2].get_value().shape, (self.classifier.n_filters,
                                                       self.classifier.n_hidden))

    def test_fit_score(self):
        new_classifier = CNNTextClassifier(seed=2, learning_rate=0.1, L1_reg=0.01, L2_reg=0.01)
        new_classifier.ready()
        print "Count score for not trained classifier..."
        test_score_before = self.classifier.score(self.x_test, self.y_test)
        train_score_before = self.classifier.score(self.x_train, self.y_train)
        print 'test score before fitting:', test_score_before
        print 'train score before fitting:', train_score_before

        try:
            new_classifier.fit(self.x_train, self.y_train, self.x_test, self.y_test, n_epochs=10)
        except KeyboardInterrupt:
            print "Fit Interrupt, if you really want to interrupt execution, try again"

        print "Count score..."
        test_score_after = new_classifier.score(self.x_test, self.y_test)
        train_score_after = new_classifier.score(self.x_train, self.y_train)
        print 'test score after fitting:', test_score_after
        print 'train score after fitting:', train_score_after

        self.assertTrue(test_score_before < test_score_after)

        new_params = new_classifier.get_cnn_params()
        new_state_path = "cnn_state_fit_score_test"
        print "Saving new state to '%s'..." % new_state_path
        new_classifier.save_state(new_state_path)

        print "Loading new state for classifier..."
        self.classifier.load(new_state_path)

        print "Comparing saved and loaded params..."
        loaded_params = self.classifier.get_cnn_params()
        self.assertEqual(len(new_params), len(loaded_params))
        for i, (param, loaded_param) in enumerate(zip(new_params, loaded_params)):
            if isinstance(param.get_value(), list):
                param = np.array(param.get_value())
                loaded_param = np.array(loaded_param.get_value())
            if not np.array_equal(param, loaded_param):
                '''
                print "ERROR!!!"
                print "PARAM[%d]:" % i
                print param.get_value()
                print "LOADED PARAM[%d]:" % i
                print loaded_param.get_value()
                '''
                pass

        print "Count score for loaded params..."
        loaded_score = self.classifier.score(self.x_test, self.y_test)
        print "loaded test score:", loaded_score
        self.assertEqual(test_score_after, loaded_score)


if __name__ == '__main__':
    unittest.main(testRunner=xmlrunner.XMLTestRunner(output='test-reports'))
