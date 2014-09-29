import numpy as np

from hillclimber import HillClimber
from functions import Objective
from supervised import Supervised


class Regressor(Supervised):

    def __init__(self, func, terms):
        self._func = func
        self._terms = terms
        self._params = None

    def train(self, features, labels):
        obj = Objective(features, labels, self)
        hc = HillClimber(obj, self._terms, (1.25, 0.8))
        hc.optimize(200, 50, 0.01)

    def predict(self, inp):
        y = self._func.evaluate(self._params, inp)
        return np.float_([y])

    def set_params(self, params):
        self._params = params


if __name__ == '__main__':
    import matrix
    import functions
    import sys
    fe = matrix.from_arff(sys.argv[1])
    la = matrix.from_arff(sys.argv[2])
    r = Regressor(functions.linear, fe.cols() + 1)
    r.train(fe, la)
    sse = r.measure_sse(fe, la)
    print(sse)
