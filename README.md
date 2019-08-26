# RBF-Network-Handwritten-Digits

Radial Basis Function (RBF) Network that can classify handwritten digits 0-9.  
  
## Table of Contents

- [Background](#background)
- [Architecture](#architecture)
- [RBF activation function](#RBF-activation-function)
- [Performance](#performance)
- [Getting Started](#getting-started)

## Background

Radial basis function network (or RBFN for short) is an artificial neural network that uses radial basis functions as activation functions. The output of the network is a linear combination of radial basis functions of the inputs and neuron parameters.  
  
## Architecture

Size: [1, 300, 10]  
  
- 1 input layer neurons  
Each training example is an image of 784 pixels (28 x 28) so there are 784 features for the input layer. However, since this is an RBF Network, instead of having a neuron for each feature, we have 1 neuron which propogates each training example of 784 features to the hidden layer RBF Neurons  
  
- 300 hidden layer neurons  
Depends on the number of centers *C* (or clusters *K*) chosen

- 10 output layer neurons  
Each for classifying a digit 0-9  

## RBF activation function

The RBF function used is a Gaussian function as shown:
<img src="http://www.sciweavers.org/tex2img.php?eq=e%5E%7B-%20%5Cbeta%20%5Cdot%20%20%5C%7C%20x%20-%20c%20%5C%7C%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" border="0" alt="e^{- \beta \dot  \| x - c \| " width="75" height="26" />

The centers *c* are selected based on random sampling of 300 data samples. A possible improvement is to use kNN clustering to get *K* clusters instead.  
  
Perceptron rule is used to update weights (Upgrade to stochastic gradient descent rule in future maybe)  

## Performance

The parameters used for this performance test are as below. These parameters have been fixed (hard-coded) as of now, but can be changed as you wish from the Network.py code.

| Parameters | Value |
| ---------- | ----: |
| beta       | 0.05  |
| Learning rate| 0.3 |

Training using all 60,000 examples takes around 3-4 minutes.  
**Results:** Using 60,000 training examples, this Network achieved ~89.7% accuracy on the test data.

## Getting Started

#### Requirements

- Numpy and Matplotlib

Extract the training data inside the Network.py directory.
Running the Network.py script, you get two options: Train and Predict.

### Train

Training will import the 60,000 training examples and train using that data. Finally the weights and centers (used for RBF) will be saved.

### Predict

Predict will import the 10,000 test examples and the weights (weights.npy). Using the weights, the Network will predict results and compare and finally compute the classification accuracy of the Network.

### Learning Curves

For diagnostics, other than classification accuracy, you can plot the learning curves to check how the Network is learning. The function `plotLearningCurves` is included and can be called after predicting.

### Training Data

Training Data used from MNIST Database. 60,000 training examples and 10,000 test examples.  
The data is in text files and zipped (compressed file size: 20.6 Mb).
