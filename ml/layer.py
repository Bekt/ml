import numpy as np


class Layer(object):

    def __init__(self, inputs, outputs):
        self._weights = np.empty((outputs, inputs))
        self._bias = np.empty(outputs)
        self._activation = np.empty(outputs)

    def feed(self, x):
        for ind, data in enumerate(self._weights):
            self._activation[ind] = np.tanh(np.sum(data * x) + self._bias[ind])