import numpy as np

def cross_entropy(A: np.array, Y: np.array):
    """Compute cross entropy.

    Arguments:
    ----------
    A -- activation output of last layer
    Y -- groundtruth

    Returns:
    ----------
    J -- cross-entropy cost
    """
    m = Y.shape[1]

    J = (-1 / m) * np.sum(np.multiply(Y, np.log(A)) + np.multiply(1 - Y, np.log(1 - A)))

    return np.squeeze(J)


cost_functions = {
    'cross-entropy': cross_entropy
}
