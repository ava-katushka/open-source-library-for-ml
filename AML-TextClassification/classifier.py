#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import cross_val_score
sys.path.insert(0, '../ConvolutionNeuralNetwork/TextClassifier')
from CNNTextClassifier import CNNTextClassifier
import datetime


import reuters
# from textclassifier import TextClassifier

"""
При запуске программы указывать путь к папке с коллекцией документов Reuters-21578
"""
    
data_path = './data'
rp = reuters.ReutersParser(data_path, multilabel=False)
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

'''
textClassifier = TextClassifier(base_classifiers = [SGDClassifier()])
q = textClassifier.fit(X_train, Y_train)#, rp.get_corpus("train", "topics", "title"))

predicted = textClassifier.predict(X_test)#, rp.get_corpus("test", "topics", "title"))

print textClassifier.score(X_test, Y_test)
print np.mean(cross_val_score(textClassifier, X_train, Y_train))
'''

n_out = max(Y_train) + 1

model_path = "../ConvolutionNeuralNetwork/TextClassifier/100features_40minwords_10context"

cnn_text_clf = CNNTextClassifier(learning_rate=0.1, window=5, n_hidden=25, n_filters=30,
                                 n_out=n_out, word_dimension=100, seed=1,
                                 model_path=model_path, L1_reg=0.01, L2_reg=0.01)
cnn_text_clf.ready()
'''
# n_hidden = 10, n_filters = 25
new_state_path = "cnn_state_fit_score_test"
print "Loading state for classifier..."
cnn_text_clf.load(new_state_path)
print "Count score..."
print "result =", np.mean(cnn_text_clf.predict(X_test) == Y_test)
'''

print "test score before training:", cnn_text_clf.score(X_test, Y_test)

try:
    cnn_text_clf.fit(X_train, Y_train, X_test, Y_test, n_epochs=20)
except KeyboardInterrupt:
    print "Fit Interrupt, if you really want to interrupt execution, try again"

new_state_path = "cnn_state_" + datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
print "Saving state to '%s'..." % new_state_path
cnn_text_clf.save_state(new_state_path)

print "test score after training:", cnn_text_clf.score(X_test, Y_test)
