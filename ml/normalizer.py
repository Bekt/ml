import numpy as np

from unsupervised import Unsupervised


class Normalizer(Unsupervised):

    def train(self, data):
        """Computes the min and max of each column."""
        self._mins = np.nanmin(data.data, axis=0)
        self._maxs = np.nanmax(data.data, axis=0)

    def transform(self, inp):
        """Normalizes continuous features."""
        if inp.shape != self._mins.shape:
            raise ValueError('Unexpected array shape.')
        return np.float_([(inp[i] - v) / (self._maxs[i] - v)
                         for i, v in enumerate(self._mins)])

    def untransform(self, inp):
        """De-normalizes continuous values."""
        if inp.shape != self._mins.shape:
            raise ValueError('Unexpected array shape.')
        return np.float_([inp[i] * (self._maxs[i] - v) + v
                         for i, v in enumerate(self._mins)])
