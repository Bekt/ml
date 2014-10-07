import matrix
import numpy as np


from unsupervised import Unsupervised


class NomCat(Unsupervised):

    def __init__(self):
        self._vals = None
        self.template = None

    def train(self, data):
        """Decides how many dimensions are needed for each column."""
        t = self._train(data)
        self._vals = t[0]
        self.template = t[1]

    @staticmethod
    def dry_train(data):
        """Returns the template matrix that would be produced
        if train() were called."""
        nc = NomCat()
        return nc._train(data)[1]

    def _train(self, data):
        """Returns a (vals, template) tuple."""
        vals = np.full(data.cols(), 1, dtype=int)
        for i in range(data.cols()):
            if data.is_nominal(i):
                vals[i] = len(data.attributes[i][1])
        total = int(np.sum(vals))
        template = matrix.empty_matrix(0, total)
        return vals, template

    def transform(self, inp):
        """Re-represent each nominal attribute with a
        categorical distribution of continuous values."""
        if self._vals.shape != inp.shape:
            raise ValueError('Unexpected size.')
        res = np.empty(self.template.cols())
        resi = 0
        for i in range(len(inp)):
            if self._vals[i] == 1:
                res[resi] = inp[i]
                resi += 1
            else:
                res[resi:resi+self._vals[i]] = 0
                if inp[i] != np.nan:
                    if inp[i] >= self._vals[i]:
                        raise ValueError('Value out of range')
                    res[resi+int(inp[i])] = 1
                resi += self._vals[i]
        return res

    def untransform(self, inp):
        """Re-encode categorical distributions as
        nominal values by finding the mode."""
        if self.template.cols() != inp.size:
            raise ValueError('Unexpected size. exp: %d, act: %d' % (self.template.cols(), inp.size))
        res = np.empty(self._vals.size)
        resi = 0
        for ind, val in enumerate(self._vals):
            if val == 1:
                res[ind] = inp[resi]
                resi += 1
            else:
                s = resi
                maxind = 0
                resi += 1
                for j in range(1, self._vals[ind]):
                    if inp[resi] > inp[s + maxind]:
                        maxind = resi - s
                    resi += 1
                res[ind] = maxind
        return res
