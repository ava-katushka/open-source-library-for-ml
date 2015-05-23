
import sys
import xmlrunner
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics

sys.path.append("./AML-TextClassification")

import reuters
from textclassifier import TextClassifier


class TextClassifierTest(unittest.TestCase):

    def setUp(self):
        data_path = "./AML-TextClassification/data"
        rp = reuters.ReutersParser(data_path, multilabel = False)
        rp.parse()
        self.X_train = rp.get_corpus("train", "topics", "data")

        self.Y_train = rp.get_corpus("train", "topics", "target")

        self.X_test = rp.get_corpus("test", "topics", "data")

        self.Y_test = rp.get_corpus("test", "topics", "target")


    def test_base_quality(self):

        sgdc = SGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1,
        eta0=0.0, fit_intercept=True, l1_ratio=0.15,
        learning_rate='optimal', loss='log', n_iter=5, n_jobs=1,
        penalty='l2', power_t=0.5, random_state=None, shuffle=True,
        verbose=0, warm_start=False)

        textClassifier = TextClassifier(base_classifiers =[sgdc, LinearSVC()], multilabel = False)
        textClassifier.fit(self.X_train, self.Y_train)
        predicted = textClassifier.predict(self.X_test)
        base_quality = 0.6
        self.assertGreaterEqual(metrics.precision_score(self.Y_test, predicted, average = 'micro'), base_quality)


    def test_advanced_quality(self):

        sgdc = SGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1,
        eta0=0.0, fit_intercept=True, l1_ratio=0.15,
        learning_rate='optimal', loss='log', n_iter=5, n_jobs=1,
        penalty='l2', power_t=0.5, random_state=None, shuffle=True,
        verbose=0, warm_start=False)

        textClassifier = TextClassifier(base_classifiers =[sgdc, LinearSVC()], multilabel = False)
        textClassifier.fit(self.X_train, self.Y_train)
        predicted = textClassifier.predict(self.X_test)
        advanced_quality = 0.8
        self.assertGreaterEqual(metrics.precision_score(self.Y_test, predicted, average = 'micro'), advanced_quality)


if __name__ == '__main__':
    unittest.main(testRunner=xmlrunner.XMLTestRunner(output='test-reports'))
    
   
