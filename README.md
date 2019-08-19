# RBF-Network-Handwritten-Digits

Radial Basis Function (RBF) Network that can classify handwritten digits 0-9.  
  
## Background

Radial basis function network (or RBFN for short) is an artificial neural network that uses radial basis functions as activation functions. The output of the network is a linear combination of radial basis functions of the inputs and neuron parameters.  
  
## Architecture

Size: [784, 300, 10]  
  
* 784 input layer neurons  
Each training example is an image of 784 pixels (28 x 28) so there are 784 features for the input layer.  
  
* 300 hidden layer neurons  
Depends on the number of clusters *K*

* 10 output layer neurons  
Each for classifying a digit 0-9  

RBF functions is used where the centers are selected based on random sampling of 300 data samples.  
  
Perceptron rule is used to update weights (Upgrade to stochastic gradient descent rule in future maybe)  
  
Training Data used from MNIST Database.
