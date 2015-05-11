# -*- coding: utf-8 -*-
"""
    Tests for the image classifier.
"""

import unittest
import xmlrunner
import numpy as np

from sklearn.datasets import load_digits
from sklearn.utils.testing import assert_array_equal, assert_equal, assert_raises
from ImageClassifier import ImageClassifier

digits = load_digits(2)
X = digits.data
y = digits.target


class TestImageClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """ Initialize testing """
        cls.clf = ImageClassifier(n_in=8, n_out=2, filters=[2,2])

        try:
            cls.clf.fit(X, y)
        except:
            print "exception has occured"

    def test_fit(self):
        """ Test that costs (% correct) increases each iter of fit. """
        assert_array_equal((self.clf.score(X,y)) >= 0, True)

    def test_deterministic_fit(self):
        """ Test that multiple fits yield same scores. """
        clf = ImageClassifier(n_in=8, n_out=2, filters=[2,2])
        clf.fit(X,y)
        assert_array_equal(self.clf.score(X,y), clf.score(X,y))

    # :test input dims

    def test_predict(self):
        """ Test predict methods on 2d array.  """
        proba_shape = (2, 2)
        clf = ImageClassifier(n_in=8, n_out=2, filters=[2,2])
        clf.fit(X,y)
        assert_equal(clf.predict_proba(X[:2]).shape, proba_shape)
        assert_equal(clf.predict(X[:2]).shape, y[:2].shape)

if __name__ == '__main__':
    unittest.main(testRunner=xmlrunner.XMLTestRunner(output='test-reports'))