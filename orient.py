#!/usr/bin/env python
#==============================================================================
# k 3 -> Accuracy 69.0 % | Time 139
# k 4 -> Accuracy 70.0 % | Time 169
# k 5 -> Accuracy 69.0 % | Time 215
# k 6 -> Accuracy 70.0 % | Time 181
# k 7 -> Accuracy 70.0 % | Time 207
# k 8 -> Accuracy 69.0 % | Time 273
# k 9 -> Accuracy 71.0 % | Time 328
# k 10 -> Accuracy 70.0 % | Time 266
#==============================================================================

from __future__ import division
import numpy as np
import sys
from numpy.linalg import norm
import cPickle as pickle
import timeit
from collections import Counter
from heapq import nsmallest

class NeuralNet(object):
    def __init__(self): pass

    def train(self, X_train, y_train): pass

    def test(self, X_test, y_test): pass

    def activation(self): pass

    def set_parameters(self): pass


class AdaBoost(object):
    def __init__(self): pass

    def train(self, X_train, y_train): pass

    def test(self, X_test, y_test): pass


class KNN(object):
    def __init__(self, k=5):
        self.k = k

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def test(self, X_test, y_test):
        correct = 0
        for X_ins, y_ins in zip(X_test, y_test):
            if y_ins == self.predict(X_ins):
                correct += 1
        return round(correct/len(X_test), 2)*100

    def predict(self, p):
        k_nearest_neighbours = self.nearest_neighbours(p)
        class_count = Counter(k_nearest_neighbours)
        return class_count.most_common()[0][0]

    def nearest_neighbours(self, p):
        distances = norm(self.X_train - p, axis=1)
        neighbors = zip(distances, self.y_train)
        k_nearest = nsmallest(self.k, neighbors, key=lambda x: x[0])
        return map(lambda x:x[1], k_nearest)


def read_file(fname):
    X = np.loadtxt(fname, usecols=range(2, 194), dtype=int)
    y = np.loadtxt(fname, usecols=1, dtype=int)#.reshape(len(X), 1)
    return X/255, y

if __name__ == "__main__":
    tic_start = timeit.default_timer()
#    task, fname, model_file, model = sys.argv[1:]
#    task, fname, model_file, model = "train train-data.txt knn.txt nearest".split()
    task, fname, model_file, model = "test test-data.txt knn.txt nearest".split()

    print "Reading data from", fname, "..."
    tic = timeit.default_timer()
    X, y = read_file(fname)
    toc = timeit.default_timer()
    print "Time taken", int(toc - tic), "seconds"

    if task == "train":
        print "Training", model
        tic = timeit.default_timer()

        if model == "nearest":
            knn = KNN()
            knn.train(X, y)
            pickle.dump(knn, open(model_file, "wb"))
        elif model == "adaboost":
            pass
        elif model == "nnet":
            pass
        else:
            pass

        toc = timeit.default_timer()
        print "Time taken", int(toc - tic), "seconds"

    else:
        print "Testing", model
        tic = timeit.default_timer()

        if model == "nearest":
            knn = pickle.load(open(model_file, "rb"))
            score = knn.test(X, y)
        elif model == "adaboost":
            pass
        elif model == "nnet":
            pass
        else:
            pass
        print "Accuracy", score, "%"
        toc = timeit.default_timer()
        print "Time taken", int(toc - tic), "seconds"

    toc_end = timeit.default_timer()
    print "Total time", int(toc_end - tic_start), "seconds"