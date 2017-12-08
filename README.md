# Assignment 4: Image classification

## Data Description

- `photo_id`: a photo ID for the image
- `correct_orientation`: 0, 90, 180, 270 which is the orientation label
- features: 192 dimensional feature vectors, where each feature vector represent either one of the red, green, blue pixel value in range 0-255
- `train-data.txt`: 36,976 training images
- `test-data.txt`: 943 test images

```
[hankjang@silo hankjang-maloop-shynaras-a4]$ wc -l train-data.txt test-data.txt
   36976 train-data.txt
     943 test-data.txt
   37919 total
```

## Classifier1: `nearest`

1. Description of how we formulated the problem including precisely defining the abstractions

k-nearest neighbor algorithm simply finds k nearest samples of a specific object, then assigns a label to the object based on the majority vote. In short, labels of the k nearest neighbors determines the label of the object.
    
There were some issues when using k-nearest neighbor. If small k was used, the classification was vulnerable to noises. However, if large k was used, it included many points from other classes. Hence, we experimented with many different values of `k`. Details are explained in part 3.    

2. Brief description of how program works

__Training__

First, we train the neural network model using the training set. The parameter sets used in the model are specified in the `orient.py`, and the trained model is saved in `knn.txt`

```
./orient.py train train-data.txt knn.txt nearest
```

__Testing__

Next, we test the test dataset using the trained model which is saved in `knn.txt`.

```
python test test-data.txt knn.txt nearest
```

3. Discussion of any problems, assumptions, simplification, and/or disign decisions made

As discussed above, If small k was used, the classification was vulnerable to noises. However, if large k was used, it included many points from other classes. Hence, we experimented with different k's in (3, 4, ..., 11). The best test accuracy was 71%, achived with k=9.
```
('Accuracy', 71.0, '%')
('Time taken', 100, 'seconds')
```

We also tried to reduce the dimension of the data to see whether accuracy goes up or not. We tried many different dimension of reducing the dataset, and found that using around 30 different eigenvectors worked fairly well. As you can see from the result below, the accuracy was about the same, but since the dimension was reduced a lot (previous 192 features, now 30 eigenvectors) the time needed for classification reduced even if you include the procedure of PCA. This is because kNN loops over the whole data points to find the k nearest neighboring points.
```
('Accuracy', 71.0, '%')
('Time taken', 32, 'seconds')
```

One interesting thing we've noticed was that using only blue pixels alone gave a similar result with using all features together. Here's a result of using only the blue pixels for training the model and testing using only the blue pixels. We got the following result from using k=5, usign only the blue pixels.
```
('Accuracy', 71.0, '%')
('Time taken', 40, 'seconds')
```

We've also tried another dimensional reduction algorithm Non-negative Matrix Factorization ("NMF") for experiments. Other than using additional algorithms, we've tried to apply PCA or NMF for each of the color pixels. In other words, we implemented three PCA procedures, one per each color pixel when processing the data. However, we could get the test accuracy to get over 71%.

## Classifier2: `adaboost`

1. Description of how we formulated the problem including precisely defining the abstractions

AdaBoost algorithm is an ensemble method that utilizes a base learning algorithm, then generate many other weak learners to later be used in majority voting for the final classification. AdaBoost is simple, has solid theoretical foundation and performs well when tested in many different domains.
    
AdaBoost first assigns equal weights to training data. AdaBoost calls the base learning algorithm to the data set and the distribution of the weights, then generate a base (weak) learner `h`. After being tested by training examples, the weights get updated; if there are incorrectly classified examples, the weights would increase. From these, another weak learner is generated. This procedure is repeated for `T` times, and the final classification is done by majority vote from `T` different learners.

In this problem, we used simple decision stumps that compares one entry in teh image matrix to another. There were 192 possible combinations to generate random pairs to try. Details are explained in part 3.

2. Brief description of how program works

__Training__

First, we train the neural network model using the training set. The parameter sets used in the model are specified in the `orient.py`, and the trained model is saved in `adaboost.txt`

```
python orient.py train train-data.txt adaboost.txt adaboost
```

__Testing__

Next, we test the test dataset using the trained model which is saved in `adaboost.txt`.

```
python -i orient.py test test-data.txt adaboost.txt adaboost
```

3. Discussion of any problems, assumptions, simplification, and/or disign decisions made

In this problem, we had to deal with a multi-class problem. In order to accomplish this, we decomposed the multi-class task to a series of binary tasks. Which means is that, we did series of one vs one classification such as (0 vs 90), (0 vs 180), (0 vs 270), (90 vs 180), (90 vs 270), and (180 vs 270). Now we have 6 sets of training data.

For each training set, we take a majority vote. Let x represent a row. Let x[4] represent 4th variable in row x and x[8] represent 8th variable in row x. If x[4] - x[8] >=0, then we label a new variable as "Positive". Otherwise, we labeled that as "Negative". For instance, out of 700 rows (based on new variable in "Positive"), lets say, 650 of them were 0 degrees. In this case we have 650 rights, and 50 wrongs. Then we assigned "Positive Class" as 0 degrees. 

This works similarly with labels with "Negative". For instance, out of 900 rows (based on new variable in "Negative"), 800 of them were 90 degrees. Hence, we got 800 rights, and 100 wrongs. In the previous cases, total rights are 650 + 800 = 1450 and total wrongs are 50 + 100 = 150. From these, initial error would be 150/1600.

The weights are initialized as 1/N. As explained above, whenever we get the correct answer, we decresed weights, and otherwise increase the weights. We then normalize the weights, so the weights of misclassified items increase. Then we move to the next decision stump.

In this way of implementation, we got around 66 to 70 percent accuracy.


## Classifier3: `nnet`

1. Description of how we formulated the problem including precisely defining the abstractions

Due to the complexity of the structure of the neural network, we first designed the classifier to work on one hidden layer and `relu` for all layers and sigmoid for the last. We tried many different combinations of number of neurons on the hidden layer and varied `alpha.`

Then, we modified the structure to make it work on different activations as well as number of layers. We've experimented the classifer by varying alpha, number of neurons, and activation function. To prevent Neural Network from overfitting the training set, we've tried diminishing value of alpha as iterations increase, and introduced a regularization term lambda. Details are described in section 3.

2. Brief description of how program works

__Training__

First, we train the neural network model using the training set. The parameter sets used in the model are specified in the `orient.py`, and the trained model is saved in `nnet.txt`

```
./orient.py train train-data.txt nnet.txt nnet
```

__Testing__

Next, we test the test dataset using the trained model which is saved in `nnet.txt`.

```
python test test-data.txt nnet.txt nnet
```

3. Discussion of any problems, assumptions, simplification, and/or disign decisions made

 Here are the parameter sets we've tried to find a good working parameter set:

### Experiment 1: One hidden layer
- alpha: <a href="https://www.codecogs.com/eqnedit.php?latex=2^{-8},&space;2^{-7},&space;...,&space;2^0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2^{-8},&space;2^{-7},&space;...,&space;2^0" title="2^{-8}, 2^{-7}, ..., 2^0" /></a>
- number of neurons: <a href="https://www.codecogs.com/eqnedit.php?latex=2^{2},&space;2^{3},&space;...,&space;2^{6}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2^{2},&space;2^{3},&space;...,&space;2^{6}" title="2^{2}, 2^{3}, ..., 2^{6}" /></a> 
- activation function: three activation functions
  - logistic: <a href="https://www.codecogs.com/eqnedit.php?latex=f(x)&space;=&space;\frac{1}{1&plus;e^{-x}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x)&space;=&space;\frac{1}{1&plus;e^{-x}}" title="f(x) = \frac{1}{1+e^{-x}}" /></a>
  - tanh: <a href="https://www.codecogs.com/eqnedit.php?latex=f(x)&space;=&space;\frac{e^x-e^{-x}}{e^x&plus;e^{-x}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x)&space;=&space;\frac{e^x-e^{-x}}{e^x&plus;e^{-x}}" title="f(x) = \frac{e^x-e^{-x}}{e^x+e^{-x}}" /></a>
  - relu: <a href="https://www.codecogs.com/eqnedit.php?latex=f(x)&space;=&space;\max(0,&space;x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x)&space;=&space;\max(0,&space;x)" title="f(x) = \max(0, x)" /></a>
  
### Experiment 2: More hidden layers
- hidden layers with number of neurons per layer: <a href="https://www.codecogs.com/eqnedit.php?latex=(2^4,&space;2^4,&space;2^4),&space;(2^4,&space;2^4,&space;2^5),&space;...&space;(2^6,&space;2^6,&space;2^6)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(2^4,&space;2^4,&space;2^4),&space;(2^4,&space;2^4,&space;2^5),&space;...&space;(2^6,&space;2^6,&space;2^6)" title="(2^4, 2^4, 2^4), (2^4, 2^4, 2^5), ... (2^6, 2^6, 2^6)" /></a>

### Experiment 3: One hidden layer, let alpha decrease as iteration increase

From the previous experiments, we could see that alpha played a big role in the performance of the Neural Network. Hence, we decided to let alpha start with some big number, then decreased it as iterations increased. In this way, we thought the algorithm would prevent getting stuck in a local minimum (bigger alpha), and later converge into a reasonable minimum point (smaller alpha in the end). Also, we implemented a regularization term lambda that would prevent the Neural Network from overfitting the training data.

We had one problem: as the iteration increase, Neural Network overfitted to the training model. Hence, we had to carefully choose the starting alpha, and experiment with diverse ratio of alpha to be decreased to prevent Neural Network to be overfitted to the training set. Surprisingly, using only one hidden layer with 193 neurons in that layer worked fairly well. Following is the parameter set we found after many experiments with parameters, hidden layers, and number of neurons. Following result is from one hidden layer (193 neurons)

```
('Test', '75.0%', 'train', '79.0%', 'log loss', 0.54874530470275185, 'alpha', 0.5, 'iterations', 2000, 'lambd', 0.1, 'layers', [192, 193, 4], 'PCA', False)
```

As we have modifeid the script to work on any kind of hidden layers and number of neurons, here's another experiment that gave us 75% accuracy on the test set. Following result is from four hidden layers (8, 6, 7, 5) neurons.
```
('Test', '75.0%', 'train', '77.0%', 'log loss', 0.58189618939350252, 'alpha', 0.02, 'iterations', 10000, 'lambd', 0.05, 'Time', 1240, 'layers', [192, 8, 6, 7, 5, 4], 'PCA', False)
```

We've tried initializing He, and even implemented dropout as an experiment. For an experiment with 4 hidden layers using dropout, the training accuracy acturally increased (not overfitting), and test accuracy increased slightly to around 76%.
