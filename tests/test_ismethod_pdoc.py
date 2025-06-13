"""
test the pdoc module
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
from sklearn.naive_bayes import GaussianNB

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent
sys.path.extend(
    [
        os.path.join(root_dir, "src"),
    ]
)

from ismethod import pdoc


class TestPDOC(unittest.TestCase):
    """test the pdoc module"""

    def setUp(self):
        # for calc_proba_diff_oppo_cls from distribution
        # for find neighbor
        # for sample by weight
        self.X = np.array(
            [
                [-1, 1],
                [0, 1],
                [1, 1],
                [-1, 0],
                [0, 0],
                [1, 0],
                [-1, -1],
                [0, -1],
                [1, -1],
            ]
        )
        self.y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1])

        # for estiamte
        self.data = (
            np.array([1, 1, 1, -1, -1]),
            [[0, 1, 2], [1, 3], [], None, [0, 1, 2, 3]],
            [[1, 2, 2], [4, 6], None, [], np.array([1, 1, 1, 1])],
            [1, 1, 1, 1, 1],
        )
        self.proba_pos_neg = ((1, 0), (0.4, 0.6), (0.6, 0.4), (0.6, 0.4), (0.75, 0.25))
        self.proba_diff = (1, -0.2, 0.2, 0.2, 0.5)
        self.proba_diff_oppo_cls = (1, -0.2, 0.2, -0.2, -0.5)

        # for sample by weight
        self.sample_weights1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.sample_weights2 = np.array([-1, -2, -3, -4, -5, -6, -7, 8, 9])

    def test_estimate_proba(self):
        """test pdoc.estimate_proba function"""
        y, ls_neighbor_idxs, ls_neighbor_weights, shrinkage_factors = self.data
        answers = self.proba_pos_neg
        for neighbor_idxs, neighbor_weights, shrinkage_factor, answer in zip(
            ls_neighbor_idxs, ls_neighbor_weights, shrinkage_factors, answers
        ):
            proba_pos, proba_neg = pdoc.estimate_proba(
                y=y,
                neighbor_idxs=neighbor_idxs,
                neighbor_weights=neighbor_weights,
                shrinkage_factor=shrinkage_factor,
            )
            self.assertEqual(proba_pos, answer[0])
            self.assertEqual(proba_neg, answer[1])

    def test_estimate_arr_proba(self):
        """test pdoc.estimate_arr_proba function"""
        y, ls_neighbor_idxs, ls_neighbor_weights, shrinkage_factors = self.data
        answers = np.array(self.proba_pos_neg)
        answers_calc = pdoc.estimate_arr_proba(
            y=y,
            ls_neighbor_idxs=ls_neighbor_idxs,
            ls_neighbor_weights=ls_neighbor_weights,
            shrinkage_factors=shrinkage_factors,
        )
        self.assertEqual(answers_calc.shape, (5, 2))
        self.assertListEqual(answers_calc.tolist(), answers.tolist())

    def test_estimate_arr_proba_diff(self):
        """test pdoc.estimate_arr_proba_diff function"""
        y, ls_neighbor_idxs, ls_neighbor_weights, shrinkage_factors = self.data
        answers = np.array(self.proba_diff)
        answers_calc = pdoc.estimate_proba_diff(
            y=y,
            ls_neighbor_idxs=ls_neighbor_idxs,
            ls_neighbor_weights=ls_neighbor_weights,
            shrinkage_factors=shrinkage_factors,
        )

        # Prevent 0.199999999999996 == 0.2 from failing
        answers_calc = answers_calc.round(16)

        self.assertEqual(answers_calc.shape, (5,))
        self.assertListEqual(answers_calc.tolist(), answers.tolist())

    def test_estimate_proba_diff_oppo_cls(self):
        """test pdoc.estimate_proba_diff_oppo_cls function"""
        y, ls_neighbor_idxs, ls_neighbor_weights, shrinkage_factors = self.data
        arr_pdoc_true = np.array(self.proba_diff_oppo_cls)
        arr_pdoc_calc = pdoc.estimate_proba_diff_oppo_cls(
            y=y,
            ls_neighbor_idxs=ls_neighbor_idxs,
            ls_neighbor_weights=ls_neighbor_weights,
            shrinkage_factors=shrinkage_factors,
        )

        # Prevent 0.199999999999996 == 0.2 from failing
        arr_pdoc_calc = arr_pdoc_calc.round(16)

        ls_neighbor_weights = [
            np.array(neighbor_weights) for neighbor_weights in ls_neighbor_weights
        ]
        arr_pdoc_true = np.array(self.proba_diff_oppo_cls)
        arr_pdoc_calc = pdoc.estimate_proba_diff_oppo_cls(
            y=y,
            ls_neighbor_idxs=ls_neighbor_idxs,
            ls_neighbor_weights=ls_neighbor_weights,
            # shrinkage_factors=shrinkage_factors,
            use_fast=True,
        )

        # Prevent 0.199999999999996 == 0.2 from failing
        arr_pdoc_calc = arr_pdoc_calc.round(16)

        self.assertEqual(arr_pdoc_calc.shape, (5,))
        self.assertListEqual(arr_pdoc_calc.tolist(), arr_pdoc_true.tolist())

    def test_calc_pdoc(self):
        """test pdoc.calc_pdoc function"""
        y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1])
        proba_x_mid_pos = np.array([0, 0.5, 1, 0, 0.5, 1, 0, 0.5, 1])
        proba_x_mid_neg = np.array([1, 0.5, 0, 1, 0.5, 0, 1, 0.5, 0])

        pos_proba_prior = 0.5
        answers = np.array([-1, 0, 1, 1, 0, -1, -1, 0, 1])
        sols = pdoc.calc_pdoc(
            y=y,
            proba_x_mid_neg=proba_x_mid_neg,
            proba_x_mid_pos=proba_x_mid_pos,
            pos_proba_prior=pos_proba_prior,
        )
        self.assertEqual(sols.shape, (9,))
        self.assertTrue(np.allclose(sols, answers))

        pos_proba_prior = 0.8
        answers = np.array([-1, 0.6, 1, 1, -0.6, -1, -1, 0.6, 1])
        sols = pdoc.calc_pdoc(
            y=y,
            proba_x_mid_neg=proba_x_mid_neg,
            proba_x_mid_pos=proba_x_mid_pos,
            pos_proba_prior=pos_proba_prior,
        )
        self.assertEqual(sols.shape, (9,))
        self.assertTrue(np.allclose(sols, answers))

    def test_calc_pdoc_xor(self):
        X = self.X
        y = self.y
        pos_proba_prior = 0.5

        sigma = 0.1
        answers = np.array([-1, 0, 1, 0, 0, 0, 1, 0, -1])
        sols = pdoc.calc_pdoc_xor(
            X=X,
            y=y,
            sigma=sigma,
            pos_proba_prior=pos_proba_prior,
        )
        self.assertTrue(np.allclose(sols, answers))

        sigma = 1
        sols = pdoc.calc_pdoc_xor(
            X=X,
            y=y,
            sigma=sigma,
            pos_proba_prior=pos_proba_prior,
        )
        self.assertEqual(sols.shape, (9,))
        self.assertTrue(sols[0] < 0)
        self.assertTrue(sols[1] == 0)
        self.assertTrue(sols[2] > 0)
        self.assertTrue(sols[3] == 0)
        self.assertTrue(sols[4] == 0)
        self.assertTrue(sols[5] == 0)
        self.assertTrue(sols[6] > 0)
        self.assertTrue(sols[7] == 0)
        self.assertTrue(sols[8] < 0)

    def test_calc_pdoc_gaussian_1(self):
        """test pdoc.calc_pdoc_gaussian function"""
        ### 1-feature
        X = self.X[:4, 0:1]  # 1 feature, -1, 0, 1, -1
        y = self.y[:4]  # 1, 1, 1, -1
        X = np.array([[-1], [0], [1], [-1]])
        y = np.array([1, 1, 1, -1])

        # proba_density_multiply_count
        pdmc = []
        for i in range(X.shape[0]):
            pdmc.append(
                [
                    3 * norm.pdf(x=X[i], loc=1, scale=1)[0],
                    norm.pdf(x=X[i], loc=-1, scale=1)[0],
                ]
            )

        answers = np.array(
            [
                (pdmc[0][0] - pdmc[0][1]) / sum(pdmc[0]),  # y=1
                (pdmc[1][0] - pdmc[1][1]) / sum(pdmc[1]),  # y=1
                (pdmc[2][0] - pdmc[2][1]) / sum(pdmc[2]),  # y=1
                (pdmc[3][1] - pdmc[3][0]) / sum(pdmc[3]),  # y=-1
            ]
        )
        pdocs = pdoc.calc_pdoc_gaussian(
            X=X,
            y=y,
            neg_mean=-1,
            neg_cov=1,
            pos_mean=1,
            pos_cov=1,
            pos_proba_prior=0.75,
        )
        self.assertEqual(pdocs.shape, (4,))

        # round to avoid 0.9136709340400074 != 0.9136709340400073
        self.assertListEqual(pdocs.round(10).tolist(), answers.round(10).tolist())

        gnb = GaussianNB()
        # set model_odb manually
        gnb.classes_ = np.array([-1, 1])
        gnb.theta_ = np.array([[-1], [1]])
        gnb.var_ = np.array([[1], [1]])  # indepent features
        gnb.class_count_ = np.array(((y == -1).sum(), (y == 1).sum()))
        gnb.class_prior_ = gnb.class_count_ / gnb.class_count_.sum()
        answers2 = y * (gnb.predict_proba(X)[:, 1] - gnb.predict_proba(X)[:, 0])

        self.assertTrue(np.allclose(pdocs, answers2))

    def test_calc_pdoc_gaussian_2(self):
        """test pdoc.calc_pdoc_gaussian function"""
        ### 2-feature
        X = np.array([[-1, 1], [0, 1], [1, 1], [-1, 0]])
        y = np.array([1, 1, 1, -1])

        # # proba_density_multiply_count
        pdmc = []
        for i in range(X.shape[0]):
            pdmc.append(
                [
                    3 * mvn.pdf(x=X[i], mean=[1, 0], cov=1),
                    mvn.pdf(x=X[i], mean=[-1, 0], cov=1),
                ]
            )

        answers = np.array(
            [
                (pdmc[0][0] - pdmc[0][1]) / sum(pdmc[0]),
                (pdmc[1][0] - pdmc[1][1]) / sum(pdmc[1]),
                (pdmc[2][0] - pdmc[2][1]) / sum(pdmc[2]),
                (pdmc[3][1] - pdmc[3][0]) / sum(pdmc[3]),
            ]
        )
        neg_mean = [-1, 0]
        pos_mean = [1, 0]
        pdocs = pdoc.calc_pdoc_gaussian(
            X=X,
            y=y,
            neg_mean=[-1, 0],
            neg_cov=[[1, 0], [0, 1]],
            pos_mean=[1, 0],
            pos_cov=[[1, 0], [0, 1]],
            pos_proba_prior=0.75,
        )
        self.assertEqual(pdocs.shape, (X.shape[0],))

        # round to avoid 0.9136709340400074 != 0.9136709340400073
        self.assertListEqual(pdocs.round(12).tolist(), answers.round(12).tolist())

        gnb = GaussianNB()
        # set model_odb manually
        gnb.classes_ = np.array([-1, 1])
        gnb.theta_ = np.array([neg_mean, pos_mean])
        gnb.var_ = np.array([[1, 1], [1, 1]])  # indepent features
        gnb.class_count_ = np.array(((y == -1).sum(), (y == 1).sum()))
        gnb.class_prior_ = gnb.class_count_ / gnb.class_count_.sum()
        answers2 = y * (gnb.predict_proba(X)[:, 1] - gnb.predict_proba(X)[:, 0])

        self.assertTrue(np.allclose(pdocs, answers2))

    def test_score_estimate(self):
        """test pdoc.score_estimate function"""
        pdocs = list(self.proba_diff_oppo_cls)
        pdocs_estim = pdocs
        for metric in ["F1", "AUC", "Accuracy", "Precision", "Recall"]:
            score = pdoc.score_estimate(pdocs, pdocs_estim, metric=metric)
            self.assertEqual(score, 1)

        for metric in ["MSE", "MAE", "RMSE", "MAPE"]:
            score = pdoc.score_estimate(pdocs, pdocs_estim, metric=metric)
            self.assertEqual(score, 0)

    def test_get_ls_neighbor_idxs(self):
        """test pdoc.get_ls_neighbor_idxs function"""
        X = self.X
        ls_neighbor_idxs_true = [
            [1, 3],
            [0, 2],
            [1, 5],
            [0, 4],
            [1, 3],
            [2, 4],
            [3, 7],
            [4, 6],
            [5, 7],
        ]
        ls_neighbor_idxs_calc = pdoc.get_ls_neighbor_idxs(X=X, n_neighbors=2)
        for neighbor_idxs_calc, neighbor_idxs_true in zip(
            ls_neighbor_idxs_calc, ls_neighbor_idxs_true
        ):
            self.assertListEqual(neighbor_idxs_calc.tolist(), neighbor_idxs_true)

        ls_neighbor_idxs_true = [
            [1, 3, 4],
            [0, 2, 3, 4, 5],
            [1, 4, 5],
            [0, 1, 4, 6, 7],
            [0, 1, 2, 3, 5, 6, 7, 8],
            [1, 2, 4, 7, 8],
            [3, 4, 7],
            [3, 4, 5, 6, 8],
            [4, 5, 7],
        ]

        ls_neighbor_idxs_calc = pdoc.get_ls_neighbor_idxs(X=X, radius=2**0.5)
        for neighbor_idxs_calc, neighbor_idxs_true in zip(
            ls_neighbor_idxs_calc, ls_neighbor_idxs_true
        ):
            self.assertListEqual(neighbor_idxs_calc.tolist(), neighbor_idxs_true)

    def test_pdoc2weight(self):
        """test pdoc.pdoc2weight function"""
        pdocs = np.array(self.proba_diff_oppo_cls)
        self.assertListEqual(
            pdoc.pdoc2weight(pdocs, close1better=True, sumas1=False).tolist(),
            [1, 0.4, 0.6, 0.4, 0.25],
        )

        self.assertTrue(
            np.allclose(
                pdoc.pdoc2weight(pdocs, close1better=True, sumas1=True),
                np.array([1, 0.4, 0.6, 0.4, 0.25]) / 2.65,
            )
        )

        self.assertListEqual(
            pdoc.pdoc2weight(pdocs, close1better=False, sumas1=False).tolist(),
            [0, 0.8, 0.8, 0.8, 0.5],
        )

        self.assertTrue(
            np.allclose(
                pdoc.pdoc2weight(pdocs, close1better=False, sumas1=True),
                np.array([0, 0.8, 0.8, 0.8, 0.5]) / 2.9,
            )
        )

    def test_sample_by_weight(self):
        """test pdoc.sample_by_weight function"""
        X = self.X.copy()
        seed = 42
        X_sub = pdoc.sample_by_weight(X, weights=self.sample_weights1, size=5)
        self.assertEqual(X_sub.shape, (5, 2))

        X_sub = pdoc.sample_by_weight(
            X, weights=self.sample_weights2, size=5, oversampling=False
        )
        self.assertEqual(X_sub.shape, (2, 2))

        X_sub = pdoc.sample_by_weight(
            X, weights=self.sample_weights2, size=5, oversampling=True
        )
        self.assertEqual(X_sub.shape, (5, 2))

        X_sub = pdoc.sample_by_weight(
            X, weights=self.sample_weights2, size=5, oversampling=True
        )

    def test_pdoc(self):
        """test pdoc.pdoc function"""
        pdocs = np.ones(9)
        selected_idxs = pdoc.pdoc(pdocs=pdocs, thresholds=[0.9, 1], selection_rate=None)
        self.assertListEqual(selected_idxs.tolist(), [0, 1, 2, 3, 4, 5, 6, 7, 8])

        selected_idxs = pdoc.pdoc(
            np.array([1, 1, 1, 0, 0, 0, -1, -1, -1]),
            thresholds=[[0.2, 0.5], [0.9, 1]],
            selection_rate=None,
        )
        self.assertListEqual(selected_idxs.tolist(), [0, 1, 2])

        selected_idxs = pdoc.pdoc(
            np.array([1, 1, 1, 0, 0, 0, -1, -1, -1]),
            thresholds=[[-1, 0.5]],
            selection_rate=None,
        )
        self.assertListEqual(selected_idxs.tolist(), [3, 4, 5, 6, 7, 8])

        selected_idxs = pdoc.pdoc(
            np.array([1, 1, 1, 0, 0, 0, -1, -1, -1]),
            thresholds=[[0, 1]],
            selection_rate=None,
            selection_type=None,
        )
        self.assertListEqual(selected_idxs.tolist(), [0, 1, 2, 3, 4, 5])

        selected_idxs = pdoc.pdoc(
            np.array([1, 1, 1, 0, 0, 0, -1, -1, -1]),
            thresholds=[[0.1, 1]],
            selection_rate=0.5,
            selection_type="all_random",
        )
        self.assertListEqual(selected_idxs.tolist(), [0, 1, 2])

        selected_idxs = pdoc.pdoc(
            np.array([1, 1, 1, 0, 0, 0, -1, -1, -1]),
            thresholds=None,
            selection_rate=0.6,
            selection_type="all_random",
        )
        self.assertEqual(selected_idxs.shape, (6,))

        selected_idxs = pdoc.pdoc(
            np.array([1, 1, 1, 0, 0, 0, -1, -1, -1]),
            thresholds=None,
            selection_rate=0.6,
            selection_type="far_random",
        )
        self.assertListEqual(sorted(selected_idxs.tolist()), [0, 1, 2, 3, 4, 5])

        selected_idxs = pdoc.pdoc(
            np.array([1, 1, 1, 0, 0, 0, -1, -1, -1]),
            thresholds=None,
            selection_rate=0.6,
            selection_type="near_random",
        )
        self.assertListEqual(sorted(selected_idxs.tolist()), [3, 4, 5])

        selected_idxs = pdoc.pdoc(
            np.array([1, 1, 1, 0, 0, 0, -1, -1, -1]),
            thresholds=None,
            selection_rate=0.6,
            selection_type="far_filter",
        )
        self.assertListEqual(sorted(selected_idxs.tolist()), [0, 1, 2, 3, 4, 5])

    # def test_pdoc_k_nn(self):
    #     """test pdoc.pdoc_k_nn function"""

    # def test_pdoc_r_nn(self):
    #     """test pdoc.pdoc_r_nn function"""


if __name__ == "__main__":
    unittest.main()
