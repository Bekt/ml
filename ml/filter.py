import copy
import numpy as np

from supervised import Supervised


class Filter(Supervised):
    """Wraps another supervised learner.
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
        if len(features.data) != len(labels.data):
            raise ValueError('Unexpected array shape.')
        if self._filter:
            matrix = self._train_transform(features)
            self._learner.train(matrix, labels)
        else:
            matrix = self._train_transform(labels)
            self._learner.train(features, matrix)

    def predict(self, inp):
        """Makes a prediction."""
        if self._filter:
            return self._learner.predict(self._transform.transform(inp))
        else:
            res = self._learner.predict(inp)
            return self._transform.untransform(res)

    def _train_transform(self, matrix):
        self._transform.train(matrix)
        nmatrix = copy.deepcopy(self._transform.template)
        nmatrix.set_data(np.empty((matrix.rows(), nmatrix.cols())))
        for ind, row in enumerate(matrix.data):
            nmatrix.data[ind] = self._transform.transform(row)
        return nmatrix
