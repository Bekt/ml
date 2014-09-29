import numpy as np
import random

from optimizer import Optimizer


class Evolutionary(Optimizer):

    def __init__(self, obj, rows, cols):
        self.name = 'evolutionary'
        self._obj = obj
        self._populations = np.random.randn(rows, cols)
        self._prob = 0.8
        self._step = 0.01
        self._best_err = 1e308

    def _evaluate(self, cand):
        err = self._obj.evaluate(cand)
        if err < self._best_err:
            self._best_err = err
        return err

    def tournament(self):
        self._best_err = 1e308
        cand1 = random.choice(self._populations)
        cand2 = random.choice(self._populations)
        if self._evaluate(cand1) > self._evaluate(cand2):
            return cand1
        else:
            return cand2

    def crossover(self, cand):
        # random.sample() doesn't work with ndarrays :/
        parent1 = random.choice(self._populations)
        parent2 = random.choice(self._populations)
        pivot = np.random.randint(cand.size)
        cand[:pivot] = parent1[:pivot]
        cand[pivot:] = parent2[pivot:]

    def interpolate(self, cand):
        parent1 = random.choice(self._populations)
        parent2 = random.choice(self._populations)
        w = np.random.random()
        child = w * parent1 + (1 - w) * parent2
        np.copyto(cand, child)

    def mutate(self, cand):
        """Move in all directions."""
        parent = random.choice(self._populations)
        np.copyto(cand, parent)
        for i in range(cand.size):
            cand[i] += self._step * np.random.normal()

    def mix(self, cand):
        """Mixes two parents."""
        parent1 = random.choice(self._populations)
        parent2 = random.choice(self._populations)
        for i in range(cand.size):
            cand[i] = parent1[i] if not random.getrandbits(1) else parent2[i]

    def move(self, cand):
        """Mutate in one direction."""
        parent = random.choice(self._populations)
        np.copyto(cand, parent)
        ind = np.random.randint(cand.size)
        cand[ind] = self._step * np.random.normal()

    def iterate(self):
        loser = self.tournament()
        action = np.random.random()
        if action < 0.1:
            self.crossover(loser)
        elif action < 0.2:
            self.interpolate(loser)
        elif action < 0.3:
            self.mutate(loser)
        elif action < 0.4:
            self.mix(loser)
        else:
            self.move(loser)
        self._evaluate(loser)
        return self._best_err
