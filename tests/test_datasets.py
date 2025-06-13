"""
test datasets.py
"""

import os
import sys
import unittest
from pathlib import Path

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent
sys.path.extend(
    [
        os.path.join(root_dir, "src"),
    ]
)
import datasets


class TestDatasets(unittest.TestCase):
    """test datasets.py"""

    def test_load_one(self):
        datasets.load_one("gen_blobs", {"n_samples": 5000, "sigma": 0.8})

    def test_load_all(self):
        count = 0
        for ds in datasets.load_all():
            count += 1
        self.assertTrue(count == len(datasets.LOARDERS))

    def test_load_one_assign_name(self):
        ds = datasets.load_one_assign_name(
            "gen_blobs", {"n_samples": 5000, "sigma": 0.8}, "test"
        )

    def test_load_all_assign_name(self):
        count = 0
        for ds in datasets.load_all_assign_name():
            count += 1

        self.assertTrue(count == len(datasets.LOARDERS))


def _test():

    dict_dataset_funcname_kwargs = {
        "blobs_0.2": {
            "func_name": "gen_blobs",
            "kwargs": {"n_samples": 5000, "sigma": 0.2},
        },
        "blobs_0.8": {
            "func_name": "gen_blobs",
            "kwargs": {"n_samples": 5000, "sigma": 0.8},
        },
        "blobs_1.5": {
            "func_name": "gen_blobs",
            "kwargs": {"n_samples": 5000, "sigma": 1.5},
        },
        "circles_0.2": {
            "func_name": "gen_circles",
            "kwargs": {"n_samples": 5000, "sigma": 0.2},
        },
        "circles_0.8": {
            "func_name": "gen_circles",
            "kwargs": {"n_samples": 5000, "sigma": 0.8},
        },
        "circles_1.5": {
            "func_name": "gen_circles",
            "kwargs": {"n_samples": 5000, "sigma": 1.5},
        },
        "moons_0.2": {
            "func_name": "gen_moons",
            "kwargs": {"n_samples": 5000, "sigma": 0.2},
        },
        "moons_0.8": {
            "func_name": "gen_moons",
            "kwargs": {"n_samples": 5000, "sigma": 0.8},
        },
        "moons_1.5": {
            "func_name": "gen_moons",
            "kwargs": {"n_samples": 5000, "sigma": 1.5},
        },
        "xor_0.2": {
            "func_name": "gen_xor",
            "kwargs": {"n_samples": 5000, "sigma": 0.2},
        },
        "xor_0.8": {
            "func_name": "gen_xor",
            "kwargs": {"n_samples": 5000, "sigma": 0.8},
        },
        "xor_1.5": {
            "func_name": "gen_xor",
            "kwargs": {"n_samples": 5000, "sigma": 1.5},
        },
        "normal_precise_1d_0.2": {
            "func_name": "gen_normal_precise_1d",
            "kwargs": {"n_samples": 5000, "sigma": 0.2},
        },
        "normal_precise_1d_0.8": {
            "func_name": "gen_normal_precise_1d",
            "kwargs": {"n_samples": 5000, "sigma": 0.8},
        },
        "normal_precise_1d_1.5": {
            "func_name": "gen_normal_precise_1d",
            "kwargs": {"n_samples": 5000, "sigma": 1.5},
        },
    }

    for ds in datasets.load_all_assign_name(dict_dataset_funcname_kwargs):
        print(ds)
        print(ds.kwargs)

    dss = list(datasets.load_all_assign_name(dict_dataset_funcname_kwargs))

    datasets.draw_scatter(dss, draw_in_one=True)

    # pdoc_k_nn = partial(pdoc.pdoc_k_nn, n_neighbors=5)
    # def pdoc_(X, y):
    #     y_ = format_y_binary(y)
    #     ls_neighbor_idxs = pdoc.get_ls_neighbor_idxs(X=X, n_neighbors=5)
    #     arr_pdoc = pdoc.calc_proba_diff_oppo_cls(
    #         y=y_, ls_neighbor_idxs=ls_neighbor_idxs
    #     )
    #     return arr_pdoc

    # draw_scatter(dss, pdoc_calc_func=pdoc_, draw_in_one=False)


# def _draw_pdoc(dss: list[Dataset] = None):
#     if dss is None:
#         dss = list(load_all_assign_name())

#     for ds in dss:
#         df, y_col = ds.df, ds.y_col
#         X, y = df[df.columns[df.columns != y_col]], df[y_col]
#         y_ = format_y_binary(y)
#         ls_neighbor_idxs = pdoc.get_ls_neighbor_idxs(X=X, n_neighbors=10)
#         arr_pdoc = pdoc.estimate_proba_diff_oppo_cls(
#             y=y_, ls_neighbor_idxs=ls_neighbor_idxs
#         )
#         df["pdoc"] = arr_pdoc
#         _, ax = plt.subplots(figsize=(8, 6))
#         sns.histplot(data=df, x="pdoc", hue=y_col, ax=ax)
#         ax.set_title(dataset.name)
#         plt.show()

if __name__ == "__main__":
    # unittest.main()
    _test()
