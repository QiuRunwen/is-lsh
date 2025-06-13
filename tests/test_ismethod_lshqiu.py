import os
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import stats

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent
sys.path.extend(
    [
        os.path.join(root_dir, "src"),
    ]
)

from ismethod import lshqiu


class TestLSHqiu(unittest.TestCase):
    def test_calc_p_collision(self):
        # https://www.wolframalpha.com/
        # integrate e^(-t^2/2)(1-t) dt from t=0 to 1
        answer = (
            pow(2 / np.pi, 0.5)
            * 0.4621550516047822267771041552712255146683324205786842160031264403
        )
        p_collis = lshqiu.calc_p_collision(dist=1, w=1)
        self.assertEqual(p_collis, answer)

        # integrate e^(-t^2/8)(1-t) dt from t=0 to 1
        answer = (
            pow(1 / (2 * np.pi), 0.5)
            * 0.4898380482581500440244163771226092139961080410054415270702510561
        )
        p_collis = lshqiu.calc_p_collision(dist=2, w=1)
        self.assertEqual(p_collis, answer)

    def test_calc_exp_collis(self):
        w = 1
        dist = 1
        t = 5
        L = 10
        answer = (
            pow(2 / np.pi, 0.5)
            * 0.4621550516047822267771041552712255146683324205786842160031264403
        ) ** t * L
        exp_collis = lshqiu.calc_exp_collis(w=w, t=t, L=L, dist=dist)
        self.assertEqual(exp_collis, answer)

    def test_lsh_2stable(self):
        # test lsh_2stable
        X = np.random.randn(100, 2)
        w, t, L = 1, 5, 10
        seed = 0
        hashtables = lshqiu.lsh_2stable(X, w=w, t=t, L=L, seed=seed)
        self.assertEqual(len(hashtables), L)
        self.assertEqual(len(hashtables[0]), X.shape[0])
        self.assertEqual(hashtables[0][0].shape, (t,))

        hashtables2 = lshqiu.lsh_2stable(X, w=w, t=t, L=L, seed=seed)
        for i in range(L):
            self.assertTrue(np.all(hashtables[i] == hashtables2[i]))

    def test_lsh_2stable_2(self):
        X = np.array([[1, 1], [1, 2]])
        dist = 1
        w, t, L = 1.5, 2, 500
        hashtables = lshqiu.lsh_2stable(X, w=w, t=t, L=L)

        proba_collis_hx = lshqiu.calc_p_collision(dist=dist, w=w)
        proba_collis_gx = lshqiu.calc_p_collision(dist=dist, w=w) ** t
        # exp_point_collis = lsh_qiu.calc_exp_collis(w=w, t=t, L=L, dist=1)

        count_collis_hx = 0
        count_collis_gx = 0
        # count_point_collis = 0

        for i in range(L):
            if (hashtables[i][0] == hashtables[i][1]).all():
                count_collis_gx += 1
            for j in range(t):
                if hashtables[i][0][j] == hashtables[i][1][j]:
                    count_collis_hx += 1

        avail_collis_hx = t * L
        avail_collis_gx = L

        exp_collis_hx = avail_collis_hx * proba_collis_hx
        exp_collis_gx = avail_collis_gx * proba_collis_gx

        # range_exp_collis_hx = (exp_collis_hx * 0.9, exp_collis_hx * 1.1)
        # range_exp_collis_gx = (exp_collis_gx * 0.9, exp_collis_gx * 1.1)

        print(f"{avail_collis_hx=}, {avail_collis_gx=}")
        print(f"{exp_collis_hx=}, {exp_collis_gx=}")
        print(f"{count_collis_hx=}, {count_collis_gx=}")
        # print(f"{range_exp_collis_hx=}, {range_exp_collis_gx=}")

        # self.assertGreaterEqual(count_collis_hx, range_exp_collis_hx[0])
        # self.assertLessEqual(count_collis_hx, range_exp_collis_hx[1])

        # self.assertGreaterEqual(count_collis_gx, range_exp_collis_gx[0])
        # self.assertLessEqual(count_collis_gx, range_exp_collis_gx[1])

        # The occurrence of collisions can be regarded as a binomial distribution
        # for h(x), B(t*L, p_collis)
        # for g(x), B(L, p_collis^t)
        # https://en.wikipedia.org/wiki/Binomial_distribution
        # suppose estimation error alpha=0.05
        # then we can caculate the confidence intervals
        # then the range of the probability of collisions is

        estimate_proba_collis_hx = count_collis_hx / avail_collis_hx
        estimate_proba_collis_gx = count_collis_gx / avail_collis_gx

        alpha = 0.01
        z = stats.norm.ppf(1 - alpha / 2)
        range_proba_collis_hx = (
            estimate_proba_collis_hx
            - z
            * np.sqrt(
                estimate_proba_collis_hx
                * (1 - estimate_proba_collis_hx)
                / avail_collis_hx
            ),
            estimate_proba_collis_hx
            + z
            * np.sqrt(
                estimate_proba_collis_hx
                * (1 - estimate_proba_collis_hx)
                / avail_collis_hx
            ),
        )
        range_proba_collis_gx = (
            estimate_proba_collis_gx
            - z
            * np.sqrt(
                estimate_proba_collis_gx
                * (1 - estimate_proba_collis_gx)
                / avail_collis_gx
            ),
            estimate_proba_collis_gx
            + z
            * np.sqrt(
                estimate_proba_collis_gx
                * (1 - estimate_proba_collis_gx)
                / avail_collis_gx
            ),
        )

        print(f"{proba_collis_hx=}, {proba_collis_gx=}")
        print(f"{range_proba_collis_hx=}, {range_proba_collis_gx=}")
        print(f"{estimate_proba_collis_hx=}, {estimate_proba_collis_gx=}")

        self.assertGreaterEqual(estimate_proba_collis_hx, range_proba_collis_hx[0])
        self.assertLessEqual(estimate_proba_collis_hx, range_proba_collis_hx[1])

        self.assertGreaterEqual(estimate_proba_collis_gx, range_proba_collis_gx[0])
        self.assertLessEqual(estimate_proba_collis_gx, range_proba_collis_gx[1])

    def test_convert_hashtable(self):
        # test convert_hashtable
        # X = np.random.randn(100, 2)
        # X = np.array([[1, 2], [1, 3], [2,4], [500, 600]])
        # w, t, L = 1, 2, 10
        # hashtables = lsh_qiu.lsh_2stable(X, w=w, t=t, L=L)

        # 0号数据和1号数据在1号表中发生碰撞
        hashtables = [
            np.array([[1, 2], [1, 3], [2, 4], [500, 600]]),
            np.array([[1, 2], [1, 2], [1, 3], [500, 600]]),
        ]
        t = hashtables[0][0].shape[0]
        L = len(hashtables)
        datasize = len(hashtables[0])

        ls_dict, ls_buckets = lshqiu.convert_hashtable(hashtables)
        self.assertEqual(len(ls_dict), L)
        self.assertEqual(len(ls_dict[0]), 4)  # 0号表中的桶数
        self.assertEqual(len(ls_dict[1]), 3)  # 1号表中的桶数

        self.assertEqual(len(ls_buckets[0]), datasize)  # 0号表中每个数据对应一个桶
        self.assertEqual(len(ls_buckets[1]), datasize)

        self.assertListEqual(ls_buckets[0][0], [0])  # 0号表的0号数据对应的桶
        self.assertListEqual(ls_buckets[0][1], [1])  # 0号表的1号数据对应的桶

        self.assertListEqual(ls_buckets[1][0], [0, 1])  # 1号表的0号数据对应的桶
        self.assertListEqual(ls_buckets[1][1], [0, 1])  # 1号表的1号数据对应的桶

        self.assertIs(
            ls_buckets[1][0], ls_buckets[1][1]
        )  # 1号表的0号数据和1号数据对应的桶是同一个桶

        # self.assertEqual(len(ls_buckets[0][0]), 0)

    def test_count_idx_in_bucket(self):
        ls_buckets = [[[0], [1], [2], [3]], [[0, 1], [0, 1], [2], [3]]]

        datasize = len(ls_buckets[0])
        # L = len(ls_buckets)

        ls_d_idx_count = lshqiu.count_idx_in_bucket(ls_buckets=ls_buckets)

        self.assertEqual(len(ls_d_idx_count), datasize)

        self.assertEqual(len(ls_d_idx_count[0]), 1)
        self.assertEqual(len(ls_d_idx_count[1]), 1)
        self.assertEqual(len(ls_d_idx_count[2]), 0)
        self.assertEqual(len(ls_d_idx_count[3]), 0)

        self.assertEqual(ls_d_idx_count[0].get(1), 1)
        self.assertEqual(ls_d_idx_count[1].get(0), 1)

    def test_lsh_2stable(self):
        X = np.random.randn(100, 2)
        w, t, L = 1, 5, 10
        seed = 0
        hashtables = lshqiu.lsh_2stable(X, w=w, t=t, L=L, seed=seed)
        self.assertEqual(len(hashtables), L)
        self.assertEqual(len(hashtables[0]), X.shape[0])
        self.assertEqual(hashtables[0][0].shape, (t,))

        hashtables2 = lshqiu.lsh_2stable(X, w=w, t=t, L=L, seed=seed)
        for i in range(L):
            self.assertTrue(np.all(hashtables[i] == hashtables2[i]))


if __name__ == "__main__":
    unittest.main()
