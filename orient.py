#!/usr/bin/env python

### KNN  ######================================================================
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
import sys
import timeit
import numpy as np
from numpy.linalg import norm
import cPickle
from collections import Counter
from heapq import nsmallest

np.random.seed(3)


class NeuralNet(object):
    def __init__(self, alpha=0.01, iterations=100, layers_dims=[192, 16, 4]):
        self.set_parameters(alpha, iterations, layers_dims)
        
    def set_parameters(self, alpha, iterations, layer_dims): 
        self.alpha = alpha
        self.iterations = iterations
        self.layer_dims = layer_dims
        self.initialize_parameters(layer_dims)
    
    def initialize_parameters(self, layer_dims):
        self.parameters = {}
        for l in range(1, len(layer_dims)):
            self.parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
            self.parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
    
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))    
            
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def linear_fwd_activation(self, A_prev, W, b, activation):
        Z = np.dot(W, A_prev) + b
        if activation == "sigmoid":
            A = self.sigmoid(Z)
        elif activation == "relu":
            A = self.relu(Z)
        cache = ((A_prev, W, b), Z)
        return A, cache 

    def forward_propogation(self, X):
        caches = []
        L = len(self.layer_dims) - 1
        
        A = X
        for l in range(1, L):
            A_prev = A
            W = self.parameters["W" + str(l)]
            b = self.parameters["b" + str(l)] 
            A, cache = self.linear_fwd_activation(A_prev, W, b, "relu")
            caches.append(cache)
            
        W = self.parameters["W" + str(L)]
        b = self.parameters["b" + str(L)]
        AL, cache = self.linear_fwd_activation(A, W, b, "sigmoid")
        caches.append(cache)
        
        return AL, caches

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = (-1/m)* np.sum( np.multiply(Y, np.log(AL)) + np.multiply(1-Y, np.log(1-AL)) )
        cost = np.squeeze(cost)
        return cost
        
    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]
        
        dW = (1/m) * np.dot(dZ, A_prev.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        
        return dA_prev, dW, db
    
    def linear_activation_backward(self, dA, cache, activation):
        ((A_prev, W, b), Z) = cache 
        
        if activation == "relu":
            dZ = np.array(dA, copy=True)
            dZ[Z <= 0] = 0
            
        elif activation == "sigmoid":
            t = 1/(1+np.exp(-Z))
            dZ = dA * t * (1-t)        
        
        dA_prev, dW, db = self.linear_backward(dZ, (A_prev, W, b))
        
        return dA_prev, dW, db
        
    def backpropogation(self, AL, Y, caches): 
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
        
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        
        cache = caches[L-1]
        grads["dA"+str(L)], grads["dW"+str(L)], grads["db"+str(L)] = self.linear_activation_backward(dAL, cache, "sigmoid")
      
        for l in range(L-1, 0, -1):
            cache = caches[l-1]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA"+str(l+1)], cache, "relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l)] = dW_temp
            grads["db" + str(l)] = db_temp
            
        return grads

    def update_parameters(self, grads):
        L = len(self.layer_dims) - 1
        for l in range(1, L+1):
            self.parameters["W"+str(l)] -= self.alpha * grads["dW"+str(l)]
            self.parameters["b"+str(l)] -= self.alpha * grads["db"+str(l)]    

    def train(self, X_train, y_train):
        X = X_train
        Y = y_train
        
        for i in range(self.iterations):
            AL, caches = self.forward_propogation(X)
#            print "Forward Propogation Complete"
            
            cost = self.compute_cost(AL, Y)
#            print "Cost Computed"
            
            grads = self.backpropogation(AL, Y, caches)
#            print "Backpropogation Complete"
#            print grads.keys()
            
            self.update_parameters(grads)
#            print "Parameters Updated"

            if i%10 == 0:
                print "Iteration", i, "-> Cost", cost
    
    def test(self, X_test, y_test):
        X_t = X_test
        Y_t = y_test

        self.forward_propogation(X_t)


class AdaBoost(object):
    def __init__(self, k = 10):
        self. k = k

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
        return round(correct/len(X_test), 2) * 100

    def predict(self, p):
        k_nearest_neighbours = self.nearest_neighbours(p)
        class_count = Counter(k_nearest_neighbours)
        return class_count.most_common()[0][0]

    def nearest_neighbours(self, p):
        distances = norm(self.X_train - p, axis=1)
        neighbors = zip(distances, self.y_train)
        k_nearest = nsmallest(self.k, neighbors, key=lambda x: x[0])
        return map(lambda x: x[1], k_nearest)



def read_file(fname):
    X = np.loadtxt(fname, usecols=range(2, 194), dtype=int)
    y = np.loadtxt(fname, usecols=1, dtype=int)  # .reshape(len(X), 1)
    return X/255, y

def transform_Y_for_NN(Y):
#    Y = Y.reshape(len(Y), 1)
    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()
    lb.fit(Y)
    return lb

if __name__ == "__main__":
#    task, fname, model_file, model = sys.argv[1:]
#    task, fname, model_file, model = "train train-data.txt knn.txt nearest".split()
#    task, fname, model_file, model = "test test-data.txt knn.txt nearest".split()
    task, fname, model_file, model = "train train-data.txt nnet.txt nnet".split()
#    task, fname, model_file, model = "test test-data.txt knn.txt nearest".split()
    
    print ("Reading data from", fname, "...")
    X, y = read_file(fname)

    if task == "train":
        print ("Training", model, "model...")
        tic = timeit.default_timer()

        if model == "nearest":
            knn = KNN(k=7)
            knn.train(X, y)
            models = (knn)
            
        elif model == "adaboost":
            pass
        elif model == "nnet":
            lb = transform_Y_for_NN(y)
            Y_lb = lb.transform(y)
            nnet = NeuralNet()
            nnet.train(X.T, Y_lb.T)
            models = (lb, nnet)
        else:
            pass
        
        cPickle.dump(models, open(model_file, "wb"), protocol=2)
        toc = timeit.default_timer()
        print ("Time taken", int(toc - tic), "seconds")

    else:
        print ("Testing", model, "model...")
        tic = timeit.default_timer()

        models = cPickle.load(open(model_file, "rb"))
        
        if model == "nearest":
            knn = models
            score = knn.test(X, y)
        elif model == "adaboost":
            pass
        elif model == "nnet":
            lb, nnet = models
            Y_lb = lb.transform(y)
            score = nnet.test(X.T, Y_lb.T)
            
        else:
            pass
        
        print ("Accuracy", score, "%")
        toc = timeit.default_timer()
        print ("Time taken", int(toc - tic), "seconds")
