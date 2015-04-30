#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics
from  sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import VarianceThreshold


import reuters
from textclassifier import TextClassifier

"""
При запуске программы указывать путь к папке с коллекцией документов Reuters-21578
"""
    
data_path = sys.argv[1]
rp = reuters.ReutersParser(data_path, multilabel = False)
"""
считываем и "парсим" данные
"""
rp.parse()
        

"""
Train sample
"""
X_train = rp.get_corpus("train", "topics", "data")

#selector = VarianceThreshold()


Y_train = rp.get_corpus("train", "topics", "target")
#Y_train = selector.fit_transform(Y_train)

"""
Test sample
"""
X_test = rp.get_corpus("test", "topics", "data")

Y_test = rp.get_corpus("test", "topics", "target")
#Y_test = selector.transform(Y_test)

print "OK"
#sel = VarianceThreshold()
#Y_test = sel.fit_transform(Y_test)

sgdc = SGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1,
        eta0=0.0, fit_intercept=True, l1_ratio=0.15,
        learning_rate='optimal', loss='log', n_iter=5, n_jobs=1,
        penalty='l2', power_t=0.5, random_state=None, shuffle=True,
        verbose=0, warm_start=False)


textClassifier = TextClassifier()
#base_classifiers =[OneVsRestClassifier(sgdc)]
textClassifier.fit(X_train, Y_train)

predicted = textClassifier.predict(X_test)
#predicted = sel.transform(predicted)

base = 1;
"""
for i in range(len(predicted)):
    #p = [line[i] for line in predicted]
    #y = [line[i] for line in Y_test]
    #print  len(p)
    #m = metrics.roc_auc_score(y, p)
    print predicted[i], Y_test[i]
    print "___________________---"
    #if (m < base):
     #   base = m
"""

print "TextClassifier"

print np.mean(predicted == Y_test)
print "precision: ",  metrics.precision_score(Y_test, predicted, average = 'micro')
print "recall: ", metrics.recall_score(Y_test, predicted, average = 'micro')
print "f1: ", metrics.f1_score(Y_test, predicted, average = 'micro')


