import os
import sys
import unittest
from pathlib import Path

import numpy
import pandas as pd

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent
sys.path.extend(
    [
        os.path.join(root_dir, "src"),
    ]
)
import preprocess


class TestPreprocess(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            {"x1": [1, 2, 3], "x2": [4, 5, 6], "x3": ["a", "b", "c"], "y": [1, 0, 1]}
        )
        self.df["x3"] = self.df["x3"].astype("category")
        self.y_col = "y"

        return super().setUp()

    def test_get_num_cat_cols(self):
        (
            num_cols,
            cat_cols,
            cat_is_num_cols,
            cat_miss_lable_cols,
        ) = preprocess.get_num_cat_cols(self.df, self.y_col)

        self.assertListEqual(num_cols, ["x1", "x2"])
        self.assertListEqual(cat_cols, ["x3"])
        self.assertListEqual(cat_is_num_cols, [])
        self.assertListEqual(cat_miss_lable_cols, [])

    def test_format_df(self):
        df, y_col = self.df, self.y_col
        X, y, num_cols, cat_cols = preprocess._format_df(df, y_col)
        self.assertListEqual(X.columns.tolist(), ["x1", "x2", "x3"])
        self.assertListEqual(y.tolist(), [1, 0, 1])
        self.assertListEqual(num_cols, ["x1", "x2"])
        self.assertListEqual(cat_cols, ["x3"])

    def test_split_impute_encode_scale(self):
        df, y_col = self.df, self.y_col
        X_train, X_test, y_train, y_test = preprocess.split_impute_encode_scale(
            df=df, y_col=y_col, test_size=0.5, seed=0
        )
        self.assertEqual(X_train.shape, (1, 3))
        self.assertEqual(X_test.shape, (2, 3))

        self.assertEqual(y_train.shape, (1,))
        self.assertEqual(y_test.shape, (2,))


if __name__ == "__main__":
    unittest.main()
