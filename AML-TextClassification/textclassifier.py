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
from  sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import VarianceThreshold



class TextClassifier:

    def __init__(self, base_classifiers = [SGDClassifier()], scoring = "auc", multilabel = True, with_titles = False ):
        """
        Parameters:
            base_classifiers: должны иметь методы fit, predict
            используются при классификации, подбирается лучшая комбинация

            scoring: precision, recall, f1
            указывает способ подбора лучшего классификатора

            multilabel: boolean
        """
        self.base_classifiers = base_classifiers
        self.best_classifier = base_classifiers[0]
        self.count_vect = CountVectorizer(decode_error='ignore')
        self.tfidf_transformer = TfidfTransformer()
        self.is_multilabel = multilabel
        self.with_titles = with_titles

    def __feature_selection(self, text_data):
        """
        Return sparse matrix of text features
        """

        X = self.count_vect.fit_transform(text_data)
        X_tfidf = self.tfidf_transformer.fit_transform(X)
        return X_tfidf

    def __transform_features(self, text_data):
        X = self.count_vect.transform(text_data)
        X_tfidf = self.tfidf_transformer.transform(X)
        return X_tfidf

    def fit(self, X, y, titles = None):
        best_quality = 0.0
        if (self.with_titles):
            X_train = [X[i] + ' ' + titles[i] for i in range(len(X))]
        else:
            X_train = X
        X_features = self.__feature_selection(X_train)
        if (self.is_multilabel):
            """
            remove target features, that are equal in all objects
            """
            self.selector = VarianceThreshold()
            Y = self.selector.fit_transform(y)
            self.best_classifier = OneVsRestClassifier(self.best_classifier)

        else: 
            for classifier in self.base_classifiers:
                new_quality = np.mean(cross_val_score(classifier, X_features, np.array(y)))
                if (new_quality > best_quality):
                    best_quality = new_quality
                    self.best_classifier = classifier
            Y = y


        self.best_classifier.fit(X_features, Y)
        if (self.is_multilabel):
            return self.selector.get_support()

    def predict(self, X, titles = None):
        if (self.with_titles):
            X_train = [X[i] + ' ' + titles[i] for i in range(len(X))]
        else:
            X_train = X
        X_features = self.__transform_features(X_train)
        return self.best_classifier.predict(X_features)

    def predict_proba(self, X):
        X_features = self.__transform_features(X)
        return self.best_classifier.predict_proba(X_features)

    def score(self, predicted, y_true):
        if (self.is_multilabel):
            Y = self.selector.transform(y_true)
        else:
            Y = y_true
        print np.mean(predicted == Y)
        print "precision: ",  metrics.precision_score(Y, predicted, average = 'micro')
        print "recall: ", metrics.recall_score(Y, predicted, average = 'micro')
        print "f1: ", metrics.f1_score(Y, predicted, average = 'micro')

    def get_params(self, deep=True):
            return {"base_classifiers": self.base_classifiers,
                "best_classifier": self.best_classifier,
                "count_vect": self.count_vect,
                "tfidf_transformer": self.tfidf_transformer,
                "is_multilabel": self.is_multilabel,
                "with_titles": self.with_titles}

