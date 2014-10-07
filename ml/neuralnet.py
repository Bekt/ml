import numpy as np
import time
import os

from functions import Objective
from supervised import Supervised


class NeuralNet(Supervised):

    def __init__(self, epochs=50):
        self._layers = []
        self._epochs = epochs

    def train(self, features, labels):
        assert(features.data.shape[0] == labels.data.shape[0])
        indices = np.arange(features.data.shape[0])
        for i in range(self._epochs):
            np.random.shuffle(indices)
            for ind in indices:
                self.forward(features.data[ind])
                self.backpropagate(labels.data[ind])
                self.update(features.data[ind])

    def predict(self, inp):
        return self.forward(inp)

    def forward(self, input_row):
        if not self._layers:
            raise ValueError('Layers is empty.')
        out = input_row
        for layer in self._layers:
            layer.feed(out)
            out = layer._output
        return out

    def backpropagate(self, output_row):
        ind = len(self._layers) - 1
        self._layers[ind].compute_blame(output_row)
        prev = self._layers[ind]
        while ind > 0:
            ind -= 1
            self._layers[ind].backprop_blame(prev)

    def update(self, inp):
        for la in self._layers:
            # Updates both weights and bias.
            la.update_weights(inp)
            inp = la._output

