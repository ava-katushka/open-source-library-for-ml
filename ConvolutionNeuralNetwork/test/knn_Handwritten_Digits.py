from sklearn.cross_validation import StratifiedKFold
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def score_accuracy(estimator, train_indices, test_indices, X_train, Y_train):
    estimator.fit(X_train[train_indices], Y_train[train_indices])
    return estimator.score(X_train[test_indices], Y_train[test_indices])

data = [] * 2
label = []
i = 0
j = 0
with open("semeion.data") as input_file:
    for line in input_file:
        line = line.strip()
        data.append([])
        j = 0
        for number in line.split():
            if j < 256:
                data[i].append(float(number))
            else:
                if float(number) > 0.5:
                    label.append(j - 256)
            j += 1
        i += 1
        
X_train = np.asarray(data[0:1593])
Y_train = np.asarray(label).ravel()
        
size = X_train.shape[0]
indices = np.arange(size)
np.random.shuffle(indices)
test_indices = indices[0:size / 7]
train_indices = indices[(size / 7) * 2:size]

max_score = -1
best_k = 2
for k in range(2,21):
    
    print "Scoring " + str(k) + " neighbours"
    
    estimator = KNeighborsClassifier(algorithm = "ball_tree", n_neighbors = k, p = 2, weights = "distance")

    score = score_accuracy(estimator, train_indices, test_indices, X_train, Y_train)
    estimator = estimator.fit(X_train, Y_train)
    
    if score > max_score:
        max_score = score
        best_k = k
        best_estimator = estimator
    print "Score of " + str(k) + " neighbours is " + str(score)
    print "Best score is " + str(max_score) + " with " + str(best_k) + " neighbours"
