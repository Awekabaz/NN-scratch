import numpy as np
from typing import Tuple
from utils.derivatives import *
from utils.backward_activations import *

def get_derivative_respect_to_activation(A: np.array, Y: np.array, name: str = 'sigmoid'):
    derivative = derivatives_dict[name]

    return derivative(A, Y)

def linear_activation_backward(dA: np.array, cache: Tuple[Tuple, np.array], name: str = 'relu'):
    linear_cache, activation_cache = cache

    backward_activation = backward_activation_functions[name]

    dZ = backward_activation(dA, activation_cache)


    A_previous, W, b = linear_cache
    m = A_previous.shape[1]

    
    dW = np.dot(dZ, A_previous.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_previous = np.dot(W.T, dZ)

    return [dW, db, dA_previous]
