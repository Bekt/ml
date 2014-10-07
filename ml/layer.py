import numpy as np


learning_rate = 0.1


class Layer(object):

    def __init__(self, inputs, outputs):
        self._weights = 0.01 * np.random.randn(outputs, inputs)
        self._bias = 0.01 * np.random.randn(outputs)
        self._actf = np.tanh
        self._actf_deriv = np.empty(outputs)
        self._net = np.empty(outputs)
        self._output = np.empty(outputs)
        self._blame = np.empty(outputs)

    def feed(self, x):
        for ind, data in enumerate(self._weights):
            self._net[ind] = np.sum(data * x) + self._bias[ind]
            self._output[ind] = self._actf(self._net[ind])
            # sech^2(x)
            self._actf_deriv[ind] = 1 - self._output[ind] * self._output[ind]

    def compute_blame(self, output_row):
        for ind, val in enumerate(self._blame):
            self._blame[ind] = ((output_row[ind] - self._output[ind])
                    * self._actf_deriv[ind])

    def backprop_blame(self, conn_layer):
        assert(conn_layer._weights.shape[1] == self._weights.shape[0])
        for ind, val in enumerate(self._blame):
            self._blame[ind] = (
                    np.sum(conn_layer._weights[:,ind] * conn_layer._blame)
                    * self._actf_deriv[ind])

    def update_weights(self, x):
        for ind, val in enumerate(self._weights):
            self._weights[ind] = (val +
                    (learning_rate * self._blame[ind] * x))
        for ind, val in enumerate(self._bias):
            self._bias[ind] = (val + 
                    (learning_rate * self._blame[ind]))
