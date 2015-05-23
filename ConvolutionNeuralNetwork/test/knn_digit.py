from sklearn.cross_validation import StratifiedKFold
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def score_accuracy(estimator, cv_strategy):
    max = 0
    for train_indices, test_indices in cv_strategy:
        estimator.fit(X_train[train_indices], Y_train[train_indices])
        a = estimator.score(X_train[test_indices], Y_train[test_indices])
        if (a >= max):
            max = a
    return max

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X_train = np.asarray(train[range(1, train.shape[1])])
Y_train = np.asarray(train[[0]]).ravel()

X_test = np.asarray(test[range(0, test.shape[1])])
Y_test = np.asarray(X_test.shape[0]).ravel()

max_score = -1
best_k = 2

for k in range(2,9):
    
    print "Scoring " + str(k) + " neighbours"
    
    estimator = KNeighborsClassifier(algorithm = "ball_tree", n_neighbors = k, p = 2, weights = "distance")

    score = score_accuracy(estimator, StratifiedKFold(Y_train, n_folds = 2))
    if score > max_score:
        max_score = score
        best_k = k
        best_estimator = estimator
        
    print "Score of " + str(k) + " neighbours is " + str(score)
    print "Best score is " + str(max_score) + " with " + str(best_k) + " neighbours"

best_estimator = best_estimator.fit(X_train, Y_train)
ans = {
    "ImageId": [i + 1 for i in range(len(X_test))],
    "Label": best_estimator.predict(X_test),
}

res = pd.DataFrame(ans, columns = ['ImageId', 'Label'])
res.to_csv("result_number.csv", index = False)
print "result is ready"
