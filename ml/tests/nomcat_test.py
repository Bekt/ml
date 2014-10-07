import numpy as np
import unittest

import matrix

from nomcat import NomCat
from io import StringIO


class TestNomcat(unittest.TestCase):
    """Tests for the nomcat module."""

    def setUp(self):
        content = """
        @relation foo
        @attribute a real
        @attribute b {x,y,z}
        @data
        10,y
        20,z
        30,x
        """
        f = StringIO(content)
        d = matrix._from_arff(f)
        self.m = matrix.Matrix(d)

    def test_transform(self):
        nc = NomCat()
        nc.train(self.m)
        self.assertTrue(([10, 0, 1, 0] == nc.transform(self.m.data[0])).all())
        self.assertTrue(([20, 0, 0, 1] == nc.transform(self.m.data[1])).all())
        self.assertTrue(([30, 1, 0, 0] == nc.transform(self.m.data[2])).all())

    def test_untransform(self):
        nc = NomCat()
        nc.train(self.m)

        orig = self.m.data[0]
        trans = nc.transform(orig)
        self.assertTrue((orig == nc.untransform(trans)).all())

        orig = self.m.data[1]
        trans = nc.transform(orig)
        self.assertTrue((orig == nc.untransform(trans)).all())

        orig = self.m.data[2]
        trans = nc.transform(orig)
        self.assertTrue((orig == nc.untransform(trans)).all())

if __name__ == '__main__':
    unittest.main()
