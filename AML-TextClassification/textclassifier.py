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
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


class TextClassifier:

    def __init__(self, base_classifiers = [SGDClassifier(), LinearSVC()], scoring = "auc"):
        """
        Parameters:
            base_classifiers: должны иметь методы fit, predict
            используются при классификации, подбирается лучшая комбинация

            scoring: precision, recall, f1
            указывает способ подбора лучшего классификатора
        """
        self.base_classifiers = base_classifiers
        self.best_classifier = base_classifiers[0]
        self.count_vect = CountVectorizer(decode_error='ignore')
        self.tfidf_transformer = TfidfTransformer()

    def __feature_selection(self, text_data):
        """
        Получаем разреженную матрицу текстовых признаков
        """

        X = self.count_vect.fit_transform(text_data)
        X_tfidf = self.tfidf_transformer.fit_transform(X)
        return X_tfidf

    def __transform_features(self, text_data):
        X = self.count_vect.transform(text_data)
        X_tfidf = self.tfidf_transformer.transform(X)
        return X_tfidf

    def fit(self, X, y):
        best_quality = 0.0
        X_features = self.__feature_selection(X)
        for classifier in self.base_classifiers:
            new_quality = np.mean(cross_val_score(classifier, X_features, np.array(y)))
            if (new_quality > best_quality):
                best_quality = new_quality
                self.best_classifier = classifier
        self.best_classifier.fit(X_features, y)

    def predict(self, X):
        X_features = self.__transform_features(X)
        return self.best_classifier.predict(X_features)




