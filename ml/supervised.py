from abc import ABCMeta, abstractmethod


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
        pass

    def cross_validate(self, features, labels, fold=2, reps=5):
        """Performs n-fold cross-validation."""
        pass
