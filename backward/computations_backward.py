import numpy as np
from typing import Tuple
from utils.derivatives import *
from utils.backward_activations import *

def get_derivative_respect_to_activation(A: np.array, Y: np.array, name: str = 'sigmoid') -> np.array:
    """Function to calculate the derivative repsect to activation function.

     Arguments:
    ----------
    A -- probability vector
    Y -- ground truth vector
    name -- name of the activation function, used to get the function from utils.derivatives modiel

    Returns:
    ----------
    dA -- vector of derivatives 
    """
    derivative = derivatives_dict[name]
    dA = derivative(A, Y)
    return dA

def linear_activation_backward(dA: np.array, cache: Tuple[Tuple, np.array], name: str = 'relu') -> Tuple(np.array):
    """Helper function to compute backpropagation.
    
    Arguments:
    ----------
    dA -- activation gradients for current layer l (size of previous layer, number of datapoints)
    cache -- tuple of caches for computations (linear cache, activation cache)
    name -- name of the activation fucntion for backward prop

    Returns:
    ----------
    dW -- gradient: cost function respect to the bias (of current layer)
    db -- gradient: cost function respect to the weights (of current layer)
    dA -- gradient: cost function respect to the activation fucntion (of the previous layer)
    """
    linear_cache, activation_cache = cache

    backward_activation = backward_activation_functions[name]

    dZ = backward_activation(dA, activation_cache)

    A_previous, W, b = linear_cache
    m = A_previous.shape[1]

    dW = np.dot(dZ, A_previous.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_previous = np.dot(W.T, dZ)

    return dW, db, dA_previous
