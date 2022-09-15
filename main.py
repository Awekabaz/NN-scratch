from model_base import NN
import numpy as np
import h5py

if __name__ == '__main__':
    architecture = [100, 5, 8, 1]
    model = NN(architecture, 'random')

    print(model.parameters)