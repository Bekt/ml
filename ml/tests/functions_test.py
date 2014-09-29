import functions
import numpy as np
import unittest

def y(c, x):
    return functions.linear.evaluate(np.float_(c), np.float_(x))


class TestLinear(unittest.TestCase):

    def test_linear(self):
        inp = [0]
        coef = [0.5, 2]
        self.assertEqual(2, y(coef, inp))

        inp = [2]
        self.assertEqual(3, y(coef, inp))

        inp = [0, 2]
        coef = [2, 3, 5]
        self.assertEqual(11, y(coef, inp))

        inp = [4, 2]
        self.assertEqual(19, y(coef, inp))
