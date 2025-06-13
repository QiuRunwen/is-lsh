"""
test methods.py
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent
sys.path.extend(
    [
        os.path.join(root_dir, "src"),
    ]
)
import methods


class TestMethods(unittest.TestCase):
    """test methods.py"""

    def setUp(self):
        # prepare data
        rng = np.random.default_rng(0)
        self.X = rng.random((100, 3))
        self.y = np.array([1] * 60 + [-1] * 40)

    def test_get_funcname_kwargs(self):
        dict_funcname_kwargs = methods.get_funcname_kwargs(handle_selrate=True)
        count = 0
        for key, val in dict_funcname_kwargs.items():
            self.assertTrue(isinstance(key, str))
            self.assertTrue(isinstance(val, dict))
            count += 1

        self.assertTrue(count == len(methods.METHOD_FUNCS))

    def test_noselect(self):
        X = self.X
        y = self.y
        sub_idxs = methods.noselect(X, y)
        self.assertTrue(np.all(sub_idxs == np.arange(len(X))))

    def test_all_method(self):
        X = self.X
        y = self.y
        dict_funcname_kwargs = methods.get_funcname_kwargs(handle_selrate=True)
        for func_name, kwargs in dict_funcname_kwargs.items():
            X_sub, y_sub, time_method = methods.sample_data(X, y, func_name, kwargs)


if __name__ == "__main__":
    unittest.main()
