#!/usr/bin/env python

### KNN  ######=================================================================
# k 3  ->  Accuracy 69.0 % | Time 101
# k 4  ->  Accuracy 70.0 % | Time 117
# k 5  ->  Accuracy 69.0 % | Time 126
# k 6  ->  Accuracy 70.0 % | Time 109
# k 7  ->  Accuracy 70.0 % | Time 103
# k 8  ->  Accuracy 69.0 % | Time 112
# k 9  ->  Accuracy 71.0 % | Time 100
# k 10 ->  Accuracy 70.0 % | Time 100
#
## If using PCA before KNN  ########
# k 3  ->  Accuracy 68.0 % | Time 30
# k 4  ->  Accuracy 68.0 % | Time 39
# k 5  ->  Accuracy 68.0 % | Time 40
# k 6  ->  Accuracy 69.0 % | Time 40
# k 7  ->  Accuracy 71.0 % | Time 32
# k 8  ->  Accuracy 70.0 % | Time 31
# k 9  ->  Accuracy 69.0 % | Time 33
# k 10 ->  Accuracy 69.0 % | Time 31
# k 11 ->  Accuracy 70.0 % | Time 42
#==============================================================================

from __future__ import division
import numpy as np
from numpy.linalg import norm
import sys
import timeit
import cPickle as pickle
from collections import Counter
from heapq import nsmallest
from sklearn.decomposition import PCA


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
    task, fname, model_file, model = sys.argv[1:]
    # task, fname, model_file, model = "train train-data.txt knn.txt nearest".split()
    # task, fname, model_file, model = "test test-data.txt knn.txt nearest".split()

    print ("Reading data from", fname, "...")
    X, y = read_file(fname)

    if task == "train":
        print ("Training", model, "model...")
        tic = timeit.default_timer()

        if model == "nearest":
            pca = PCA(n_components=0.85, svd_solver="full")
            X = pca.fit_transform(X)

            knn = KNN(k=7)
            knn.train(X, y)
            pickle.dump((pca, knn), open(model_file, "wb"))
        elif model == "adaboost":
            pass
        elif model == "nnet":
            pass
        else:
            pass

        toc = timeit.default_timer()
        print ("Time taken", int(toc - tic), "seconds")

    else:
        print ("Testing", model, "model...")
        tic = timeit.default_timer()

        if model == "nearest":
            pca, knn = pickle.load(open(model_file, "rb"))
            X = pca.transform(X)
            score = knn.test(X, y)
        elif model == "adaboost":
            pass
        elif model == "nnet":
            pass
        else:
            pass
        print ("Accuracy", score, "%")
        toc = timeit.default_timer()
        print ("Time taken", int(toc - tic), "seconds")

    toc_end = timeit.default_timer()
    print ("Total time", int(toc_end - tic_start), "seconds")
