#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics
from  sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.cross_validation import cross_val_score
import pickle


import reuters
from textclassifier import TextClassifier

"""
При запуске программы указывать путь к папке с коллекцией документов Reuters-21578
"""
    
data_path = sys.argv[1]
rp = reuters.ReutersParser(data_path, multilabel = True)
"""
считываем и "парсим" данные
"""
rp.parse()
        

"""
Train sample
"""
X_train = rp.get_corpus("train", "topics", "data")

Y_train = rp.get_corpus("train", "topics", "target")

"""
Test sample
"""
X_test = rp.get_corpus("test", "topics", "data")

Y_test = rp.get_corpus("test", "topics", "target")

print "OK"

sgdc = SGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1,
        eta0=0.0, fit_intercept=True, l1_ratio=0.15,
        learning_rate='optimal', loss='log', n_iter=5, n_jobs=1,
        penalty='l2', power_t=0.5, random_state=None, shuffle=True,
        verbose=0, warm_start=False)

names = rp.get_names("topics")
filename = "names.pkl"
with open(filename, "wb") as f: 
    s = pickle.dump(names, f, protocol=2)



textClassifier = TextClassifier(base_classifiers = [SGDClassifier()])
textClassifier.fit(X_train, Y_train)#, rp.get_corpus("train", "topics", "title"))

predicted = textClassifier.predict(X_test)#, rp.get_corpus("test", "topics", "title"))

#save model to file
filename = "class.pkl"
with open(filename, "wb") as f: 
    s = pickle.dump(textClassifier, f, protocol=2)


print textClassifier.score(X_test, Y_test)
print np.mean(cross_val_score(textClassifier, X_train, Y_train))

