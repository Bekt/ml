from unsupervised import Unsupervised


class Imputer(Unsupervised):

    def train(self, data):
        """Computes the column means of continuous attributes,
        and the mode of nominal attributes."""
        pass

    def transform(self, inp):
        """Replaces missing values with the centroid value."""
        pass

    def untransform(self, inp):
        """Copy in-to-out. In other words, this is a no-op."""
        pass
