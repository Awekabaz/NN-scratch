import numpy as np
from itertools import chain
from typing import Dict, List

def generate_seq(ceil):
    return None

def flatten_vector(parameters: Dict[str, float]) -> List[float]:
    """Function used to flatten dictionary into a single list.
    """
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
    """Function used to check the implemntation of Backprop.
    Using vanilla numerical method: compute derivative of Cost Function and compare to base gradient vector.
    """
    parameters_values = flatten_vector(parameters)

    grad = flatten_vector(gradients)

    return None