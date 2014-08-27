from unsupervised import Unsupervised


class Normalizer(Unsupervised):

    def train(self, data):
        """Computes the min and max of each column."""
        pass

    def transform(self, inp):
        """Normalizes continuous features."""
        pass

    def untransform(self, inp):
        """De-normalizes continuous values."""
        pass
