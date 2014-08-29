#!/usr/bin/env python3

import matrix
import numpy as np
import unittest

from io import StringIO


class TestMatrix(unittest.TestCase):
    """Tests for the matrix module."""

    def setUp(self):
        content = """
        @relation foo
        @attribute width  numeric
        @attribute height numeric
        @attribute color  {red,green,blue,yellow,black}
        @data
        5.0,3.25,black
        4.5,3.75,?
        3.0,?,red
        """
        f = StringIO(content)
        d = matrix._from_arff(f)
        self.m = matrix.Matrix(d)

    def test_arff(self):
        self.assertEqual(3, self.m.cols())
        self.assertEqual(3, self.m.rows())
        self.assertIs(type(self.m.data), np.ndarray)
        self.assertEqual(self.m.data[0, 0], 5.0)
        self.assertEqual(self.m.data[0, 2], 4.0)
        self.assertTrue(np.isnan(self.m.data[1, 2]))
        self.assertTrue(np.isnan(self.m.data[2, 1]))

    def test_matrix(self):
        self.assertTrue(self.m.is_nominal(2))
        self.assertTrue(self.m.is_numerical(0))
        self.assertEqual('color', self.m.attr_name(2))
        self.assertAlmostEqual(sum([5, 4.5, 3]) / 3, self.m.column_mean(0))
        self.assertAlmostEqual(sum([3.25, 3.75]) / 2, self.m.column_mean(1))
        self.assertEqual(5, self.m.column_max(0))
        self.assertEqual(3.75, self.m.column_max(1))
        self.assertEqual(3.25, self.m.column_min(1))
        self.assertEqual(0, self.m.column_common(2))
