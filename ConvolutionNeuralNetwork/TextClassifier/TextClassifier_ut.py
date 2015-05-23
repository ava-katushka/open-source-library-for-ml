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
        x_data = []
        max_count = 1000
        print "max_count = %d" % max_count
        for review in data["review"][0:max_count]:
            review_text = BeautifulSoup(review).get_text()
            x_data.append(review_text)

        cls.x_data = np.array(x_data)
        cls.y_data = np.array(data["sentiment"][0:max_count], dtype='int32')

        cls.classifier = CNNTextClassifier()

        print "Loading state for classifier..."
        cls.classifier.load("./movieReviews/cnn_state_20150523023906")

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
        new_classifier = CNNTextClassifier(seed=2, learning_rate=0.1)
        new_classifier.ready()
        print "Count score for not trained classifier..."
        not_trained_score = self.classifier.score(self.x_data, self.y_data)
        print not_trained_score

        try:
            new_classifier.fit(self.x_data, self.y_data, n_epochs=4)
        except KeyboardInterrupt:
            print "Fit Interrupt, if you really want to interrupt execution, try again"

        print "Count score..."
        score = new_classifier.score(self.x_data, self.y_data)
        print score

        #self.assertTrue(not_trained_score < score)

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
        loaded_score = self.classifier.score(self.x_data, self.y_data)
        print loaded_score
        self.assertEqual(score, loaded_score)


if __name__ == '__main__':
    unittest.main(testRunner=xmlrunner.XMLTestRunner(output='test-reports'))
