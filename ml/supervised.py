import numpy as np

from abc import ABCMeta, abstractmethod


class Supervised(metaclass=ABCMeta):

    @abstractmethod
    def train(self, features, labels):
        """Train this supervised learner."""
        pass

    @abstractmethod
    def predict(self, inp):
        """Make a prediction."""
        pass

    def measure_sse(self, features, labels):
        """Returns SSE. Uses Hamming distance for nominal attributes,
        and means for continuous.
        """
        s = 0
        for i, row in enumerate(features.data):
            predictions = self.predict(row)
            if len(predictions) != labels.cols():
                raise ValueError('Predicted size not the same as columns.')
            mag = 0
            for j, pred in enumerate(predictions):
                actual = labels.data[i,j]
                if labels.is_nominal(j):
                    mag += 1 if np.isclose(actual, pred) else 0
                else:
                    diff = actual - pred
                    mag += diff * diff
            s += mag
        return s

    def cross_validate(self, features, labels, fold=2, reps=5):
        """Performs n-fold cross-validation.
        :returns double MSE.
        """
        pass
