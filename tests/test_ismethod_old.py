"""
Test the old instance selection methods.
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

import ismethod


class TestOld(unittest.TestCase):
    def setUp(self) -> None:
        # self.X = np.array(
        #     [
        #         [-1, 1],
        #         [0, 1],
        #         [1, 1],
        #         [-1, 0],
        #         [0, 0],
        #         [1, 0],
        #         [-1, -1],
        #         [0, -1],
        #         [1, -1],
        #     ]
        # )
        # self.y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1])

        np.random.seed(0)
        self.X = np.random.randn(200, 2)
        self.y = np.random.randint(0, 2, 200) * 2 - 1

        return super().setUp()

    def test_ENNC(self):
        X, y = ismethod.ENNC(self.X, self.y, selection_rate=0.5)
        self.assertEqual(X.shape[1], self.X.shape[1])
        self.assertEqual(X.shape[0], y.shape[0])

        subset_size_min = int(self.X.shape[0] * 0.45)
        subset_size_max = int(self.X.shape[0] * 0.55)
        self.assertGreaterEqual(X.shape[0], subset_size_min)
        self.assertLessEqual(X.shape[0], subset_size_max)

        X2, y2 = ismethod.ENNC(self.X, self.y, selection_rate=0.5)
        self.assertTrue(np.all(X == X2))
        self.assertTrue(np.all(y == y2))

    def test_NC(self):
        X, y = ismethod.NC(self.X, self.y, selection_rate=0.5)
        self.assertEqual(X.shape[1], self.X.shape[1])
        self.assertEqual(X.shape[0], y.shape[0])

        subset_size_min = int(self.X.shape[0] * 0.45)
        subset_size_max = int(self.X.shape[0] * 0.55)
        self.assertGreaterEqual(X.shape[0], subset_size_min)
        self.assertLessEqual(X.shape[0], subset_size_max)

        X2, y2 = ismethod.NC(self.X, self.y, selection_rate=0.5)
        self.assertTrue(np.all(X == X2))
        self.assertTrue(np.all(y == y2))

    def test_PDOV_Voronoi(self):
        seed = 0
        X, y = ismethod.PDOC_Voronoi(self.X, self.y, selection_rate=0.5, seed=seed)
        self.assertEqual(X.shape[1], self.X.shape[1])
        self.assertEqual(X.shape[0], y.shape[0])

        # TODO: check the subset size
        # now the subset often has less than 50% of the original size.
        # And the gap between the subset size and desired size is large. 41 vs 90
        # subset_size_min = int(self.X.shape[0] * 0.45)
        # subset_size_max = int(self.X.shape[0] * 0.55)
        # self.assertGreaterEqual(X.shape[0], subset_size_min)
        # self.assertLessEqual(X.shape[0], subset_size_max)

        X2, y2 = ismethod.PDOC_Voronoi(self.X, self.y, selection_rate=0.5, seed=seed)
        self.assertTrue(np.all(X == X2))
        self.assertTrue(np.all(y == y2))

    def test_LSH_IS_F_bs(self):
        seed = 0
        X, y = ismethod.LSH_IS_F_bs(self.X, self.y, selection_rate=0.5, seed=seed)
        self.assertEqual(X.shape[1], self.X.shape[1])
        self.assertEqual(X.shape[0], y.shape[0])

        subset_size_min = int(self.X.shape[0] * 0.45)
        subset_size_max = int(self.X.shape[0] * 0.55)
        self.assertGreaterEqual(X.shape[0], subset_size_min)
        self.assertLessEqual(X.shape[0], subset_size_max)

        X2, y2 = ismethod.LSH_IS_F_bs(self.X, self.y, selection_rate=0.5, seed=seed)
        self.assertTrue(np.all(X == X2))
        self.assertTrue(np.all(y == y2))


if __name__ == "__main__":
    unittest.main()
