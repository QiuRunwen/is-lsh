"""
test exp.py
"""

import json
import os
import shutil
import sys
import unittest
from pathlib import Path
from types import MappingProxyType

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent
sys.path.extend(
    [
        os.path.join(root_dir, "src"),
    ]
)
import exp
import exp_config


# TODO two functions below will use the same loggers, the .log is not correct
class TestExp(unittest.TestCase):
    """test exp.py"""

    def setUp(self):
        self.conf = MappingProxyType(
            {
                "dataset_func_kwargs": {
                    "blobs": {
                        "func_name": "gen_blobs",
                        "kwargs": {"n_samples": 5000, "sigma": 1.5},
                    }
                },
                "method_func_kwargs": {
                    "PDOC_kNN": {
                        "func_name": "pdoc_knn",
                        "kwargs": {"n_neighbors": 10, "selection_rate": 0.1, "seed": 0},
                    }
                },
                "model_func_kwargs": {
                    "LR": {"func_name": "LogisticRegression", "kwargs": {}}
                },
                "other_kwargs": {
                    "seeds": [0, 1, 2, 3, 4],
                    "output_dir": os.path.join(root_dir, "output/result/test_exp"),
                    "test_size": 0.2,
                    "use_cache": True,
                    "time_limit": 600,
                },
            }
        )

        self.default_conf = exp_config.gen_default_exp_conf()
        self.default_conf["dataset_func_kwargs"] = {
            "blobs": {
                "func_name": "gen_blobs",
                "kwargs": {"n_samples": 1000, "sigma": 0.8},
            }
        }
        self.default_conf["other_kwargs"]["output_dir"] = os.path.join(
            root_dir, "output/result/test_exp_default"
        )
        self.default_conf["other_kwargs"]["time_limit"] = 3
        self.default_conf["other_kwargs"]["seeds"] = [0, 1]
        self.default_conf = MappingProxyType(self.default_conf)

    def test_exp(self):
        """test exp.exp function"""
        fp = os.path.join(root_dir, "exp_conf", "exp_test.json")
        if os.path.exists(fp):
            with open(fp, mode="r", encoding="utf-8") as file:
                conf = json.load(file)
        else:
            conf = {
                "dataset_func_kwargs": {
                    "blobs": {
                        "func_name": "gen_blobs",
                        "kwargs": {"n_samples": 1000, "sigma": 1.5},
                    }
                },
                "method_func_kwargs": {
                    "PDOC_kNN": {
                        "func_name": "pdoc_knn",
                        "kwargs": {"n_neighbors": 10, "selection_rate": 0.1, "seed": 0},
                    }
                },
                "model_func_kwargs": {
                    "LR": {"func_name": "LogisticRegression", "kwargs": {}}
                },
                "other_kwargs": {
                    "seeds": [0, 1, 2, 3, 4],
                    "output_dir": "./output/result/test",
                    "test_size": 0.2,
                    "use_cache": True,
                },
            }

        dict_dataset_kwargs = conf["dataset_func_kwargs"]
        dict_method_kwargs = conf["method_func_kwargs"]
        dict_model_kwargs = conf["model_func_kwargs"]
        other_kwargs = conf["other_kwargs"]
        other_kwargs["use_cache"] = False

        other_kwargs["output_dir"] = os.path.join(
            other_kwargs["output_dir"], "test_exp"
        )
        output_dir = other_kwargs["output_dir"]

        # remove the test dir and all its contents for a clean test
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        exp.exp(
            dict_dataset_kwargs, dict_method_kwargs, dict_model_kwargs, **other_kwargs
        )

        other_kwargs["use_cache"] = True
        exp.exp(
            dict_dataset_kwargs, dict_method_kwargs, dict_model_kwargs, **other_kwargs
        )

    def test_exp_with_error(self):

        conf = self.conf.copy()
        dict_dataset_kwargs = conf["dataset_func_kwargs"]
        dict_method_kwargs = conf["method_func_kwargs"]
        dict_model_kwargs = conf["model_func_kwargs"]
        other_kwargs = conf["other_kwargs"]
        other_kwargs["use_cache"] = False

        other_kwargs["output_dir"] = os.path.join(
            other_kwargs["output_dir"], "test_exp_with_error"
        )
        output_dir = other_kwargs["output_dir"]
        other_kwargs["time_limit"] = 0.1  # # set a small value see the error situation

        # remove the test dir and all its contents for a clean test
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        exp.exp(
            dict_dataset_kwargs, dict_method_kwargs, dict_model_kwargs, **other_kwargs
        )

        other_kwargs["use_cache"] = True
        exp.exp(
            dict_dataset_kwargs, dict_method_kwargs, dict_model_kwargs, **other_kwargs
        )

    def test_reproduce(self):
        """test exp.exp function to reproduce the same result"""
        for conf in (self.conf.copy(), self.default_conf.copy()):
            dict_dataset_kwargs = conf["dataset_func_kwargs"]
            dict_method_kwargs = conf["method_func_kwargs"]
            dict_model_kwargs = conf["model_func_kwargs"]
            other_kwargs = conf["other_kwargs"]
            other_kwargs["use_cache"] = False

            other_kwargs["output_dir"] = os.path.join(
                other_kwargs["output_dir"], "reproduce1"
            )
            output_dir = other_kwargs["output_dir"]
            # remove the test dir and all its contents for a clean test
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            df1 = exp.exp(
                dict_dataset_kwargs,
                dict_method_kwargs,
                dict_model_kwargs,
                **other_kwargs,
            )

            other_kwargs["output_dir"] = other_kwargs["output_dir"] + "2"
            output_dir = other_kwargs["output_dir"]

            # remove the test dir and all its contents for a clean test
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)

            df2 = exp.exp(
                dict_dataset_kwargs,
                dict_method_kwargs,
                dict_model_kwargs,
                **other_kwargs,
            )

            if df1.shape != df2.shape:
                print(
                    "df1 and df2 have different shapes, probably causing by `time_limit`."
                    " Now compare result with same exp_id in df1 and df2."
                    " If they are the same, the result is reproducible."
                )
                print(f"df1.shape={df1.shape}, df2.shape={df2.shape}")
                exp_ids1 = df1["exp_id"].unique()
                exp_ids2 = df2["exp_id"].unique()
                if len(exp_ids1) > len(exp_ids2):
                    df1 = df1[df1["exp_id"].isin(exp_ids2)]
                else:
                    df2 = df2[df2["exp_id"].isin(exp_ids1)]

            self.assertTrue(
                (df1.set_index("exp_id")["F1"] == df2.set_index("exp_id")["F1"]).all()
            )
            cols = [
                col for col in df1.columns if col not in ("time_method", "time_model")
            ]
            self.assertTrue(df1[cols].equals(df2[cols]))


if __name__ == "__main__":
    unittest.main()
