import numpy as np

from optimizer import Optimizer


class HillClimber(Optimizer):

    def __init__(self, obj, terms, dirs):
        if not terms or terms < 1:
            raise ValueError('terms need to be >= 1')
        self.name = 'hillclimber'
        self._cur = np.zeros(terms)
        self._step = np.full(terms, 0.1)
        self._dirs = dirs

    def iterate(self):
        for ind, val in enumerate(self._cur):
            err = self._obj.evaluate(self._cur)
            orig = err
            err, v = self._reevaluate(ind, -self._dirs[0], val, err, val)
            err, v = self._reevaluate(ind, -self._dirs[1], val, err, v)
            err, v = self._reevaluate(ind, self._dirs[1], val, err, v)
            err, v = self._reevaluate(ind, self._dirs[0], val, err, v)
            self._cur[ind] = v
            if not np.isclose(v, val):
                self._step[ind] = v - val
            elif np.isclose(err, orig):
                self._step[ind] *= np.min(self._dirs)
        return err

    def _reevaluate(self, ind, cons, orig, err, val):
        self._cur[ind] = self._cur[ind] + cons * self._step[ind]
        e = self._obj.evaluate(self._cur)
        if e < err:
            err, val = e, self._cur[ind]
        self._cur[ind] = orig
        return err, val
