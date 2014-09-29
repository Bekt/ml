import numpy as np

class Objective(object):

    def __init__(self, features, labels, learner):
        self._features = features
        self._labels = labels
        self._learner = learner

    def evaluate(self, inp):
        self._learner.set_params(inp)
        return self._learner.measure_sse(self._features, self._labels)


class Linear(object):
    """Given coef (m) and inp (x) vectors,
    evaluate returns y such that:
      y = m[0]*x[0] + m[1]*x[1] ... b
    """
    def evaluate(coef, inp):
        if len(coef) != len(inp) + 1:
            raise ValueError('coef size needs to be inp size + 1.')
        s = np.sum(coef[0:-1] * inp)
        s += coef[-1]
        return s


linear = Linear
