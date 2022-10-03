import numpy as np
import configs as cfg
from typing import List, Tuple, Union, NoReturn, Optional, Dict
from forward.computations_forward import *
from backward.computations_backward import *

np.random.seed(cfg.ENV['SEED'])

class NN():

    def __init__(self, layers_dimension: List[int], initialisation_method: str = 'random'):
        """NN class initialisation function. 

        Arguments:
        ----------
        layers_dimension -- architecture is defined using this list of integers (including input layer)
        initialisation_method -- method for initialising parameters (default is 'random')
        """
        self.num_layers = len(layers_dimension)
        self.layers_dimensions = layers_dimension
        self.learning_rate = cfg.ENV['LEARNING']
        self.parameters = self.initialize_parameters(layers_dimension, initialisation_method)
        self.gradients = {}
        self.cache = []

    @staticmethod
    def initialize_parameters(layers_dimension, method) -> Dict[str, List[float]]:
        """Function to initialise parameters of deep NN. 
        Architecture is defined in layers_dimension. Eg:
        [Linear->Activation]*(L-1) -> [Linear->Activation]

        Arguments:
        ----------
        layers_dimension -- architecture is defined using this list of integers (including input layer)
        initialisation_method -- method for initialising parameters (default is 'random')

        Returns:
        ----------
        parameters -- dictionary of parameters for corresponding layer:
                    Wl: weights of l layer, matrix of shape (layers_dimension[l], layers_dimension[l-1])
                    bl: bias vector, matrix of shape (layers_dimension[l], 1)
        """
        parameters = {}
        for l in range(1, len(layers_dimension)):
            weights = np.random.randn(layers_dimension[l], layers_dimension[l-1])*cfg.ENV['DUMPING']
            bias = np.zeros((layers_dimension[l], 1))
            parameters['layer' + str(l)] = [weights, bias]

        return parameters

    def forward(self, X: np.array) -> np.array:
        """Function to implement forward propagation.

        Arguments:
        ----------
        X -- input data, matrix of shape (input size, number of datapoints)

        Returns:
        ----------
        A -- activation value of the last layer
        """
        A = X
        L = len(self.parameters)

        for l in range(1, L):
            A_previous = A 
            A, cache_t = linear_activation_forward(A_previous, 
                    self.parameters['layer'+str(l)][0], self.parameters['layer'+str(l)][1], 'relu')
            self.cache.append(cache_t)

        AL, cache_t = linear_activation_forward(A, 
                self.parameters['layer'+str(L)][0], self.parameters['layer'+str(L)][1], 'sigmoid')
        self.cache.append(cache_t)

        return AL

    def compute_cost(self, A: np.array, Y: np.array, name: str = 'cross-entropy'):
        """Compute the cost function.

        Arguments:
        ----------
        A -- activation output of last layer, matrix of shape (1, number of examples)
        Y -- ground truth values, 1, number of examples)
        name -- name of the cost function

        Returns:
        ----------
        J -- cost value
        """
        J = cost(A, Y, name)
        return J

    def backward(self, A: np.array, Y: np.array) -> NoReturn:
        """Implementation of backropagation of our NN.
        Arguments:
        ----------
        A -- activation output of last layer, matrix of shape (1, number of examples)
        Y -- ground truth values, 1, number of examples)

        Returns:
        ----------
        gradients - dictionary of the gradients of corresponding layer:
                [0]: dW, same shape as W matrix
                [1]: db, same shape as b matrix
                [2]: dA, same shape as A matrix
        """
        m = A.shape[1]
        # Y = Y.reshape(A.shape) # validation
        L = len(self.parameters)
        dA = get_derivative_respect_to_activation(A, Y,'sigmoid')

        current_cache = self.cache[L-1]
        grads = linear_activation_backward(dA, current_cache, name = "sigmoid")
        self.gradients['dW'+str(L)] = grads[0]
        self.gradients['db'+str(L)] = grads[1]
        self.gradients['dA'+str(L-1)] = grads[2]

        for l in reversed(range(L-1)):
            current_cache = self.cache[l]
            grads = linear_activation_backward(self.gradients["dA"+str(l+1)],  current_cache, name = "relu")
            self.gradients['dW'+str(l+1)] = grads[0]
            self.gradients['db'+str(l+1)] = grads[1]
            self.gradients['dA'+str(l)] = grads[2]

    def update_parameters_gradient(self) -> NoReturn:
        """Method to update parameters, gradient descent.
        """
        L = len(self.parameters)
        for l in range(L):
            self.parameters['layer'+str(l+1)][0] = self.parameters['layer'+str(l+1)][0] \
                - self.learning_rate*self.gradients['dW'+str(l+1)]

            self.parameters['layer'+str(l+1)][1] = self.parameters['layer'+str(l+1)][1] \
                - self.learning_rate*self.gradients['db'+str(l+1)]

    def predict(self, X: np.array, Y: np.array, treshold: float = 0.5) -> np.array:
        """Method to predict using updated parameters.
        Arguments:
        ----------
        X -- array of data points, shape of (input size, number of datapoints[m])
        Y -- array of ground truths, shape of (1, number of datapoints[m])

        Returns:
        ----------
       predictions -- predicted values, shape of (1, number of datapoints)

        """
        m = X.shape[1]
        L = len(self.parameters)
        predictions = np.zeros((1,m))
        
        probs = self.forward(X)

        for i in range(0, probs.shape[1]):
            if probs[0,i] > treshold:
                predictions[0,i] = 1
            else:
                predictions[0,i] = 0

        print("Accuracy: "  + str(np.sum((predictions == Y)/m)))
        return predictions

    def __str__(self):
        print('Model Architecture:')
        print('\tInput layer - {} units'.format(self.layers_dimensions[0]))
        for layer in range(1,len(self.layers_dimensions)):
            print('\tLayer_{} - {} units'.format(layer, self.layers_dimensions[layer]))