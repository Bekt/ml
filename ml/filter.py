from supervised import Supervised


class Filter(Supervised):
    """This class wraps another supervised learner.
    It applies some unsupervised operations to the data before
    presenting it to the learner."""

    def __init__(self, learner, transform, filter_inputs=False):
        """If filter_inputs is True, then it applies the transform
        to the input features only. Otherwise, it applies the transform
        the the output labels only."""
        self._learner = learner
        self._transform = transform
        self._filter = filter_inputs

    def train(self, features, labels):
        """Trains the supervised learner and the transform."""
        pass

    def predict(self, inp):
        """Makes a prediction."""
        pass
