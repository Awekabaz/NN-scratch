import numpy as np

def derivative_to_sigmoid(A, Y):
    dA = - (np.divide(Y, A) - np.divide(1 - Y, 1 - A))
    return dA

derivatives_dict = {
    'sigmoid': derivative_to_sigmoid
}