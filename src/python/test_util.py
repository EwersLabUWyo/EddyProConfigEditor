import unittest
from datetime import datetime

from util import (
    or_isinstance,
    in_range,
    compare_configs,
    compute_date_overlap,)

class TestUtil(unittest.TestCase):
    def test_or_isinstance(self):
        self.assertEqual(or_isinstance(int(1), int, float, str), True)
        self.assertEqual(or_isinstance(int(1), str, float, list), False)
        self.assertEqual(or_isinstance(int(1), int, int, int), True)

    def test_in_range(self):
        fun = in_range
        self.assertTrue(fun(0, '[-10, 10]'))
        self.assertTrue(fun(0, '[-10, 0]'))
        self.assertTrue(fun(0, '[-inf, inf]'))
        self.assertTrue(fun(0, '[0, 10]'))

        self.assertFalse(fun(0, '[-10, 0)'))
        self.assertFalse(fun(0, '[-inf, -inf]'))
        self.assertFalse(fun(0, '(0, 10]'))

        self.assertRaises(AttributeError, fun, 0, 10)
        self.assertRaises(TypeError, fun, '0', '[-10, 10]')
        self.assertRaises(AssertionError, fun, 10, ']10, 10)')
        self.assertRaises(AssertionError, fun, 10, ')10, 10)')
        self.assertRaises(AssertionError, fun, 10, '(10, 10[')
        self.assertRaises(ValueError, fun, 10, '(10, 10()')

    def test_compute_date_overlap(self):
        fun = compute_date_overlap

        self.assertRaises(
            AssertionError, 
            fun, 
            ('2019-01-01', datetime(2020, 1, 1)), 
            (datetime(2019, 1, 1), datetime(2020, 1, 1)))
        self.assertRaises(
            AssertionError, 
            fun, 
            (datetime(2020, 1, 1), '2019-01-01'), 
            (datetime(2019, 1, 1), datetime(2020, 1, 1)))
        self.assertEqual(
            fun(
                (datetime(2019, 1, 1), datetime(2020, 1, 1)), 
                (datetime(2019, 1, 1), datetime(2020, 1, 1))).days,
            365)
        self.assertEqual(
            fun(
                (datetime(2020, 1, 1), datetime(2019, 1, 1)), 
                (datetime(2020, 1, 1), datetime(2019, 1, 1))).days,
            -365)
        self.assertEqual(
            fun(
                (datetime(2019, 1, 1), datetime(2020, 1, 1)), 
                (datetime(2019, 3, 1), datetime(2019, 9, 1))).days,
            184)
        self.assertEqual(
            fun(
                (datetime(2019, 3, 1), datetime(2019, 9, 1)),
                (datetime(2019, 1, 1), datetime(2020, 1, 1))).days, 
            184)
        self.assertEqual(
            fun(
                (datetime(2019, 1, 1), datetime(2019, 9, 1)),
                (datetime(2019, 3, 1), datetime(2020, 1, 1))).days, 
            184)
    

        
        



