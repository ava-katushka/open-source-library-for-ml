#!/usr/bin/python
# -*- coding: utf-8 -*-
import reuters 
import sys
import os
import numpy as np
from glob import glob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

class TextClassifier:

    def __init__(self, base_classifiers):
        self.base_classifiers = base_classifiers
        self.best_classifier = base_classifiers[0]


    def fit(self, X, y):
        best_quality = 0.0
        for classifier in self.base_classifiers:
            classifier.fit(X, y)
            new_quality = np.mean(cross_val_score(classifier, X_train, np.array(Y_train)))
            if (new_quality > best_quality):
                best_quality = new_quality
                self.best_classifier = classifier

    def predict(self, X):
        return self.best_classifier.predict(X)




"""
При запуске программы указывать путь к папке с коллекцией документов Reuters-21578
"""

    
data_path = sys.argv[1]
rp = reuters.ReutersParser(data_path)
"""
считываем и "парсим" данные
"""
for filename in glob(os.path.join(data_path, "*.sgm")):
    with open(filename, "r") as f:
        rp.parse(f)
        

"""
Получаем разреженную матрицу текстовых признаков
"""


count_vect = CountVectorizer(decode_error='ignore')

"""
Обучающая выборка
"""
text_data = [line["body"] for line in rp.get_corpus()["train"] if len(line["orgs"]) != 0]
X_train = count_vect.fit_transform(text_data)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train)

Y_train = [line["orgs"][0] for line in rp.get_corpus()["train"] if len(line["orgs"]) != 0]


"""
Тестовая выборка
"""
test_samples = rp.get_corpus()["test"]
text_data_test = [line["body"] for line in rp.get_corpus()["test"]  if len(line["orgs"]) != 0]

X_test = count_vect.transform(text_data_test)


X_test_tfidf = tfidf_transformer.transform(X_test)

Y_test = [line["orgs"][0] for line in test_samples  if len(line["orgs"]) != 0]



def study_classifier(classifier, X_train, Y_train, X_test, Y_test):
    classifier.fit(X_train, Y_train)
    predicted = classifier.predict(X_test)
    print np.mean(predicted == Y_test) 

    print np.mean(cross_val_score(classifier, X_train, np.array(Y_train)))


"""
Простой "наивный" баейсовский классификатор
"""

print "Байес:"
study_classifier(MultinomialNB(), X_train_tfidf, Y_train, X_test_tfidf, Y_test)

"""
SVM
"""

print "SVM:"
sgdc = SGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1,
        eta0=0.0, fit_intercept=True, l1_ratio=0.15,
        learning_rate='optimal', loss='modified_huber', n_iter=5, n_jobs=1,
        penalty='l2', power_t=0.5, random_state=None, shuffle=True,
        verbose=0, warm_start=False)
study_classifier(sgdc, X_train_tfidf, Y_train, X_test_tfidf, Y_test)


"""
LinearSVC
"""

print "LinearSVC:"
study_classifier(LinearSVC(), X_train_tfidf, Y_train, X_test_tfidf, Y_test)

textClassifier = TextClassifier([sgdc, LinearSVC()])
textClassifier.fit(X_train_tfidf, Y_train)
predicted = textClassifier.predict(X_test_tfidf)

print "TextClassifier"
print np.mean(predicted == Y_test)



