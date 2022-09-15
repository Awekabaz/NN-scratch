import numpy as np

def sigmoid(linear):
    """Sigmoid activation function
    Arguments:
    ----------
    linear -- output of linear layer

    Returns:
    ----------
    A -- output of activation function
    """
    A = 1/(1+np.exp(-linear))
    
    return A, linear

def relu(linear):
    """Sigmoid activation function
    Arguments:
    ----------
    linear -- output of linear layer

    Returns:
    ----------
    A -- output of activation function
    """
    A = np.maximum(0,linear)
    
    assert(linear.shape == linear.shape)
    
    return A, linear

activation_functions = {
    'sigmoid': sigmoid,
    'relu': relu
}