# -*- coding: utf-8 -*-
import unittest
import xmlrunner
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

from TextClassifier import TextClassifier

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestTrainedTextClassifier(unittest.TestCase):
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

        cls.x_data = x_data
        cls.y_data = np.array(data["sentiment"][0:max_count], dtype='int32')

        cls.classifier = TextClassifier()

        print "Loading state for classifier..."
        cls.classifier.load("./movieReviews/cnn_state_last")

    def test_score(self):
        print "Count score..."
        score = self.classifier.score(self.x_data, self.y_data)
        self.assertTrue(score > 0.5)


if __name__ == '__main__':
    unittest.main(testRunner=xmlrunner.XMLTestRunner(output='test-reports'))
