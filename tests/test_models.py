"""
test models.py
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
import models


class TestModels(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = rng.random((100, 3))
        self.y = np.array([1] * 60 + [0] * 40)

    def test_train_model(self):
        model, time_model = models.train_model(
            X_train=self.X,
            y_train=self.y,
            func_name="LogisticRegression",
            # kwargs={"cv": 5},
        )
        self.assertTrue(isinstance(model, models.LogisticRegression))

    def test_all_models(self):
        count = 0
        for (
            fname,
            kwargs,
        ) in models.get_funcname_kwargs().items():
            model, time_model = models.train_model(
                X_train=self.X,
                y_train=self.y,
                func_name=fname,
                kwargs=kwargs,
            )
            count += 1
        self.assertTrue(count == len(models.DICT_FUNCNAME_FUNC))


if __name__ == "__main__":
    unittest.main()
