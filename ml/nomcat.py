from unsupervised import Unsupervised


class NomCat(Unsupervised):

    def train(self, data):
        """Decides how many dimensions are needed for each column."""
        pass

    def transform(self, inp):
        """Re-represent each nominal attribute with a
        categorical distribution of continuous values."""
        pass

    def untransform(self, inp):
        """Re-encode categorical distributions as
        nominal values by finding the mode."""
        pass
