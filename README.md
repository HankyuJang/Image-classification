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

## Classifier2: `adaboost`

1. Description of how we formulated the problem including precisely defining the abstractions
2. Brief description of how program works
3. Discussion of any problems, assumptions, simplification, and/or disign decisions made

## Classifier3: `nnet`

1. Description of how we formulated the problem including precisely defining the abstractions

Due to the complexity of the structure of the neural network, we first designed the classifier to work on one hidden layer and `relu` for all layers and sigmoid for the last. We tried many different combinations of number of neurons on the hidden layer and varied `alpha.`

Then, we modified the structure to make it work on different activations as well as number of layers. We've experimented the classifer by varying alpha, number of neurons, and activation function. Details are described in section 3.

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

### Experiment with one hidden layer
- alpha: <a href="https://www.codecogs.com/eqnedit.php?latex=2^{-8},&space;2^{-7},&space;...,&space;2^0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2^{-8},&space;2^{-7},&space;...,&space;2^0" title="2^{-8}, 2^{-7}, ..., 2^0" /></a>
- number of neurons: <a href="https://www.codecogs.com/eqnedit.php?latex=2^{2},&space;2^{3},&space;...,&space;2^{6}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2^{2},&space;2^{3},&space;...,&space;2^{6}" title="2^{2}, 2^{3}, ..., 2^{6}" /></a> 
- activation function: three activation functions
  - logistic: <a href="https://www.codecogs.com/eqnedit.php?latex=f(x)&space;=&space;\frac{1}{1&plus;e^{-x}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x)&space;=&space;\frac{1}{1&plus;e^{-x}}" title="f(x) = \frac{1}{1+e^{-x}}" /></a>
  - tanh: <a href="https://www.codecogs.com/eqnedit.php?latex=f(x)&space;=&space;\frac{e^x-e^{-x}}{e^x&plus;e^{-x}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x)&space;=&space;\frac{e^x-e^{-x}}{e^x&plus;e^{-x}}" title="f(x) = \frac{e^x-e^{-x}}{e^x+e^{-x}}" /></a>
  - relu: <a href="https://www.codecogs.com/eqnedit.php?latex=f(x)&space;=&space;\max(0,&space;x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x)&space;=&space;\max(0,&space;x)" title="f(x) = \max(0, x)" /></a>
  
### Experiment with more layers
- hidden layers with number of neurons per layer: <a href="https://www.codecogs.com/eqnedit.php?latex=(2^4,&space;2^4,&space;2^4),&space;(2^4,&space;2^4,&space;2^5),&space;...&space;(2^6,&space;2^6,&space;2^6)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(2^4,&space;2^4,&space;2^4),&space;(2^4,&space;2^4,&space;2^5),&space;...&space;(2^6,&space;2^6,&space;2^6)" title="(2^4, 2^4, 2^4), (2^4, 2^4, 2^5), ... (2^6, 2^6, 2^6)" /></a>
