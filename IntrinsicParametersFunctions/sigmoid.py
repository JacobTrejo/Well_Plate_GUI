import numpy as np

def sigmoid(x, scaling):
    y = 1 / (1 + np.exp(-scaling * x))
    return y