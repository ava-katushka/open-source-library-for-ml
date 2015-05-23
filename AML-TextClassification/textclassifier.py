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
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score


class TextClassifier(BaseEstimator):

    def __init__(self, base_classifiers = [SGDClassifier()]):
        """
        Parameters
        ----------
        base_classifiers: array, shape = [n_estimators], optional, default: [SGDClassifier()]
            estimators objects implementing fit and predict
            used for classification, the best combination is choosen

        scoring: string, optional, precision, recall, f1

            указывает способ подбора лучшего классификатора

        multilabel: boolean, optional, default: True

        with_titles: boolean, optional, default: False

        Attributes
        ----------
        multilabel_: boolean, optional, default: True
        with_titles_: boolean, optional, default: False

        """
        self.base_classifiers = base_classifiers

    def __feature_selection(self, text_data):
        """
        Transform data by using tf-idf 

        Parameters
        ----------
            text_data: array, shape = [n_samples]

        Returns
        -------
        sparse matrix of text features
        """
        X = self.count_vect_.fit_transform(text_data)
        X_tfidf = self.tfidf_transformer_.fit_transform(X)
        return X_tfidf

    def __transform_features(self, text_data):
        """
        Parameters
        ----------

        Returns
        -------
        """
        X = self.count_vect_.transform(text_data)
        X_tfidf = self.tfidf_transformer_.transform(X)
        return X_tfidf

    def fit(self, X, y, titles = None, multilabel = True):
        """
        Fit base_classifiers, choose the best model

        Parameters
        ----------
            X: array, shape = [n_samples]
            y: array, shape = [n_samples]
            titles: array, shape = [n_samples], optional, default: None
            multilabel: boolean, optional, default:True

        Returns
        -------
        self
        """
        self.with_titles_ = (titles != None)
        self.multilabel_ = multilabel
        self.tfidf_transformer_ = TfidfTransformer()
        self.count_vect_ = CountVectorizer(decode_error='ignore')
        self.best_classifier_ = self.base_classifiers[0]
        best_quality = 0.0
        if (self.with_titles_):
            X_train = [X[i] + ' ' + titles[i] for i in range(len(X))]
        else:
            X_train = X
        X_features = self.__feature_selection(X_train)
        if (self.multilabel_):
            """
            remove target features, that are equal in all objects
            """
            self.selector_ = VarianceThreshold()
            Y = self.selector_.fit_transform(y)
            self.best_classifier_ = OneVsRestClassifier(self.best_classifier_)
        else: 
            for classifier in self.base_classifiers:
                new_quality = np.mean(cross_val_score(classifier, X_features, np.array(y)))
                if (new_quality > best_quality):
                    best_quality = new_quality
                    self.best_classifier_ = classifier
            Y = y
        self.best_classifier_.fit(X_features, Y)
        return self

    def predict(self, X):
        """
        Parameters
        ----------

        Returns
        -------
        """
        if (self.with_titles_):
            X_train = [X[i] + ' ' + titles[i] for i in range(len(X))]
        else:
            X_train = X
        X_features = self.__transform_features(X_train)
        return self.best_classifier_.predict(X_features)

    def predict_proba(self, X):
        """
        Parameters
        ----------
        Returns
        -------
        """
        X_features = self.__transform_features(X)
        return self.best_classifier_.predict_proba(X_features)

    def score(self, X, y_true):
        """
        Parameters
        ----------
        Returns
        -------
        """
        if (self.multilabel_):
            Y = self.selector_.transform(y_true)
            return accuracy_score(Y, self.predict(X))
        else:
            return accuracy_score(Y, self.predict(X))


