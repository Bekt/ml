import time

from abc import ABCMeta, abstractmethod
from numpy.random import RandomState


class Supervised(object):

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
        for ind, row in enumerate(features.data):
            predicted = predict(row)
            actual = labels.row(ind)[0]
            if labels.is_nominal(0) and np.isclose(actual, predicted):
                s += 1
            else:
                diff = actual - predicted
                s += diff * diff
        return s

    def cross_validate(self, features, labels, fold=2, reps=5):
        """Performs n-fold cross-validation.
        :returns double MSE.
        """
        pass
