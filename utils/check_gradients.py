import numpy as np
from itertools import chain

def generate_seq(ceil):
    None

def flatten_vector(parameters):
    flattened = list(chain(*parameters.values()))
    count = 0

    for i in flattened:
        new_vector = np.reshape(i, (-1, 1))
        if count == 0:
            final_vector = new_vector
        else:
            final_vector = np.concatenate((final_vector, new_vector), axis=0)
        count = count + 1

    return final_vector

def check_backprop(parameters, gradients, X, Y, epsilon = 1e-7):
    parameters_values = flatten_vector(parameters)

    grad = flatten_vector(gradients)

    return None