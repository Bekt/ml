from abc import ABCMeta, abstractmethod


class Unsupervised(metaclass=ABCMeta):

    @abstractmethod
    def train(self, data):
        """Trains this unsupervised learner."""
        pass

    @abstractmethod
    def transform(self, inp):
        """Transform a single instance."""
        pass

    @abstractmethod
    def untransform(self, inp):
        """Untransform a single instance."""
        pass
