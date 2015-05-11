import numpy, os, cPickle
from ImageClassifier import ImageClassifier


def load_mnist():
    path = '..'
    data = cPickle.load(open(os.path.join(path,'mnist.pkl'), 'r'))
    return data

(train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = load_mnist()
print test_X.shape
clf = ImageClassifier(n_in=28,n_out=10)
clf.fit(train_X, train_Y)
results = clf.predict(test_X)
print numpy.mean(results==test_Y)