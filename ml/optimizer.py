import numpy as np

from abc import ABCMeta, abstractmethod


class Optimizer(metaclass=ABCMeta):

    def __init__(self, objective):
        if not objective:
            raise ValueError('objective must be provided')
        self._obj = objective

    @abstractmethod
    def iterate(self):
        """Returns a double."""
        pass

    def optimize(self, burns, window, threshold):
        for i in range(burns):
            self.iterate()
        err = self.iterate()
        while True:
            prev = err
            for i in range(window):
                self.iterate()
            err = self.iterate()
            if (prev - err) / prev < threshold or np.isclose(err, 0):
                break
        return err
