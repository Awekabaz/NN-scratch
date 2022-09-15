import numpy as np

def derivative_to_sigmoid(A, Y):
    return - (np.divide(Y, A) - np.divide(1 - Y, 1 - A))

derivatives_dict = {
    'sigmoid': derivative_to_sigmoid
}