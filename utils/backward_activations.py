import numpy as np

def sigmoid_backward(dA, cache):
    """Backward propagation - single SIGMOID unit.

    Arguments:
    ----------
    dA -- activation gradient
    cache -- linear cache, namely Z

    Returns:
    ----------
    dZ -- Gradient of the cost function with respect to Z
    """
    Z = cache
    
    sigmoid = 1/(1+np.exp(-Z))
    dZ = dA * sigmoid * (1-sigmoid)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def relu_backward(dA, cache):
    """Backward propagation - single RELU unit.

    Arguments:
    ----------
    dA -- activation gradient
    cache -- linear cache, namely Z

    Returns:
    ----------
    dZ -- Gradient of the cost function with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    
    return dZ

backward_activation_functions = {
    'sigmoid': sigmoid_backward,
    'relu': relu_backward
}