from __future__ import division
from sklearn.neighbors import KNeighborsClassifier as KNN_sklearn
import timeit
import numpy as np
from orient import KNN

def read_file(fname):
    X = np.loadtxt(fname, usecols=range(2, 194), dtype=int)
    y = np.loadtxt(fname, usecols=1, dtype=int)
    return X/255, y

X, y = read_file("train-data.txt")
Xt, yt = read_file("test-data.txt")

tic = timeit.default_timer()

knn = KNN_sklearn()
knn.fit(X, y)
score = knn.score(Xt, yt)
print round(score, 2)*100, "%"

toc = timeit.default_timer()
print "Time", int(toc - tic)

print "==============="

scores = []
for k in range(3, 11):
    tic = timeit.default_timer()

    knn = KNN(k)
    knn.train(X, y)
    score = knn.test(Xt, yt)

    toc = timeit.default_timer()

    print "k", k, "-> Accuracy", score, "%", "| Time", int(toc - tic)


    scores.append(score)