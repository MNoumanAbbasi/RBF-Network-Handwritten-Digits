# RBF-Network-Handwritten-Digits

Radial Basis Function (RBF) Network that can classify handwritten digits 0-9.  
  
## Table of Contents

- [Background](#background)
- [Architecture](#architecture)
- [Support](#support)
- [Contributing](#contributing)

## Background

Radial basis function network (or RBFN for short) is an artificial neural network that uses radial basis functions as activation functions. The output of the network is a linear combination of radial basis functions of the inputs and neuron parameters.  
  
## Architecture

Size: [1, 300, 10]  
  
- 1 input layer neurons  
Each training example is an image of 784 pixels (28 x 28) so there are 784 features for the input layer. However, since this is an RBF Network, instead of having a neuron for each feature, we have 1 neuron which propogates each training example of 784 features to the hidden layer RBF Neurons  
  
- 300 hidden layer neurons  
Depends on the number of clusters *K*

- 10 output layer neurons  
Each for classifying a digit 0-9  

## RBF activation function

The RBF function used is a Gaussian function as shown is used where the centers are selected based on random sampling of 300 data samples.  
  
Perceptron rule is used to update weights (Upgrade to stochastic gradient descent rule in future maybe)  
  
Training Data used from MNIST Database.
