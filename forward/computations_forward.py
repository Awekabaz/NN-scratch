import numpy as np
from utils.forward_activations import *
from utils.costs import *

def linear_activation_forward(A_previous: np.array, W: np.array, b: np.array, name: str = 'relu'):
    """Helper function to compute linear->activation.
    
    Arguments:
    ----------
    A -- activations form previous layer, matrix of shape (size of previous layer, number of datapoints)
    W -- weights, matrix of shape (size of current layer, size of previous layer)
    b -- bias vector, matrix of form (size of current layer, 1)
    activation -- activation function to be used

    Returns:
    ----------
    output -- result of the activation function
    cache -- tuple containing (linear_cache, activation cache), used for efficient computation of backpropagation
    """
    activation_function = activation_functions[name]

    Z = np.dot(W, A_previous) + b
    linear_cache = (A_previous, W, b)

    A, activation_cache = activation_function(Z)

    cache = (linear_cache, activation_cache)

    return A, cache

def cost(A, Y, name: str = 'cross-entropy'):
    cost_function = cost_functions[name]

    return cost_function(A, Y)