# -*- coding: utf-8 -*-
"""
Created on Thu May  5 16:36:51 2022

@author: iwenc, QiuRunwen
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy

from . import dataset as ds

###############################################################################
# 用 LSH 求 (R,c)-NN 的算法相关的一些估算公式
###############################################################################

#
# HarPeled2012
# @article{v008a014,
#  author = {Har-Peled, Sariel and Indyk, Piotr and Motwani, Rajeev},
#  title = {Approximate Nearest Neighbor: Towards Removing the Curse of Dimensionality},
#  year = {2012},
#  pages = {321--350},
#  doi = {10.4086/toc.2012.v008a014},
#  publisher = {Theory of Computing},
#  journal = {Theory of Computing},
#  volume = {8},
#  number = {14},
#  URL = {https://theoryofcomputing.org/articles/v008a014},
# }


#
# MDatar2004
# @inproceedings{10.1145/997817.997857,
# author = {Datar, Mayur and Immorlica, Nicole and Indyk, Piotr and Mirrokni, Vahab S.},
# title = {Locality-Sensitive Hashing Scheme Based on p-Stable Distributions},
# year = {2004},
# isbn = {1581138857},
# publisher = {Association for Computing Machinery},
# address = {New York, NY, USA},
# url = {https://doi.org/10.1145/997817.997857},
# doi = {10.1145/997817.997857},
# abstract = {We present a novel Locality-Sensitive Hashing scheme for the Approximate Nearest Neighbor Problem under lp norm, based on p-stable distributions.Our scheme improves the running time of the earlier algorithm for the case of the lp norm. It also yields the first known provably efficient approximate NN algorithm for the case p<1. We also show that the algorithm finds the exact near neigbhor in O(log n) time for data satisfying certain "bounded growth" condition.Unlike earlier schemes, our LSH scheme works directly on points in the Euclidean space without embeddings. Consequently, the resulting query time bound is free of large factors and is simple and easy to implement. Our experiments (on synthetic data sets) show that the our data structure is up to 40 times faster than kd-tree.},
# booktitle = {Proceedings of the Twentieth Annual Symposium on Computational Geometry},
# pages = {253–262},
# numpages = {10},
# keywords = {p-stable distributions, approximate nearest neighbor, sublinear algorithm, locally sensitive hashing},
# location = {Brooklyn, New York, USA},
# series = {SCG '04}
# }
#
def compute_p2(r_over_c):
    r"""
    Compute Pr(h_{a,b}(v1) == h_{a,b}(v2)), where h_{a,b}(x) = \lfloor (ax + b) / r \rfloor
    Where x is a d-dimensional vector, a is a d-dimensiontal random vector with elements from
    2-stable distribution Gaussian N(0,1).

    Since the formula only depends on r/c, we implemented it as a single variable function

    Parameters
    ----------
    r_over_c : float
        r/c, where c = ||v1 - v2||_2, r is bucket width

    Returns
    -------
    float in [0,1]
        The formula p_2 in line 2 left column pp. 256 of MDatar2004

    """
    const = 2 / np.sqrt(2 * np.pi)
    b = 1 - np.exp(-(r_over_c**2) / 2)
    return 1 - 2 * scipy.stats.norm.cdf(-r_over_c) - const / r_over_c * b


def compute_p2_c_r(c, r):
    r"""
    Compute Pr(h_{a,b}(v1) == h_{a,b}(v2)), where h_{a,b}(x) = \lfloor (ax + b) / r \rfloor
    Where x is a d-dimensional vector, a is a d-dimensiontal random vector with elements from
    2-stable distribution Gaussian N(0,1).

    ----------
    c : float,
        c = ||v1 - v2||_2
    r : float
        bucket width in LSH function.

    Returns
    -------
    float in [0,1]

        The formula p_2 in line 2 left column pp. 256 of MDatar2004

    """
    return compute_p2(r / c)


def compute_rho(r1, r2, r):
    """
    Give a (R, cR, p1, p2)-sensitive hashing function,
    compute rho = (log (1/p1)) / (log (1/p2))

    Parameters
    ----------
    R : float
        Positive
    c : float
        Positive
    r : float
        bucket width

    Returns
    -------
    float
        rho
    """
    p1 = compute_p2_c_r(c=r1, r=r)
    p2 = compute_p2_c_r(c=r2, r=r)
    # return np.log(1/p1) / np.log(1/p2)
    return np.log(p1) / np.log(p2)


def plot_1b_MDatar():
    c_lst = np.linspace(1, 10, 181)[1:]
    r_lst = np.linspace(0, 30, 301)[1:]
    best_rho_lst = []
    for c in c_lst:
        r1 = 1.0
        rho = [compute_rho(r1, c * r1, r) for r in r_lst]
        best_rho = min(rho)
        best_rho_lst.append(best_rho)
    plt.plot(c_lst, best_rho_lst, label="rho")
    plt.plot(c_lst, 1 / c_lst, label="1/c")
    plt.yticks(np.arange(0, 1.01, 0.1))
    plt.legend()
    plt.show()


# plot_1b_MDatar()


# plot fig 2 b)
# Compare our figure and Figure 2(b) on page 258 of MDatar2004, we can confirm
# that M. Datar set r_1 = 1.0 and r_2 = c * r_1, where c is in (R, c)-NN problem
# defined by HarPeled2012
def plot_2b_MDatar(r1=1.0, max_r=20):
    x = np.linspace(0, max_r, 101)[1:]
    for c in [1.1, 1.5, 2.5, 5, 10]:
        plt.plot(x, [compute_rho(r1, c * r1, r) for r in x], label=f"{c=}")
    plt.yticks(np.arange(0, 1.01, 0.1))
    plt.title(f"{r1=}")
    plt.ylabel("rho")
    plt.xlabel("r")
    plt.xlim(0, max_r + 5)
    plt.legend()
    plt.show()


# plot_2b_MDatar()


# introducing a new variable alpha, rewrite r = alpha r1
# rho = log f(r/r1) / log f(r/r2) = log f(alpha) / log f(alpha/c)
# 我们的 alpha 就相当于 plot_2b_MDatar 中的 r
# 这时无论 r1 是多大， 给定一个 c 最优的 alpha 不变
def plot_2b(r1=1.0, max_r=20):
    x = np.linspace(0, max_r, 101)[1:]
    for c in [1.1, 1.5, 1.7, 2, 2.5, 3, 5, 10]:
        plt.plot(x, [compute_rho(r1, c * r1, r * r1) for r in x], label=f"{c=}")
    plt.yticks(np.arange(0, 1.01, 0.1))
    plt.title(f"{r1=}")
    plt.ylabel("rho")
    plt.xlabel("r/r1")
    plt.xlim(0, max_r + 5)
    plt.legend()
    plt.show()


# plot_2b(r1=1.0) # 与 plot_2b_MDatar() 一样
# plot_2b(r1=2.0) # 与 plot_2b_MDatar() 一样


def compute_rho2(c, alpha):
    """
    We first select r1, then set r = alpha r1 and set r2 = c r1

    Parameters
    ----------
    c : TYPE
        DESCRIPTION.
    alpha : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    return np.log(compute_p2(alpha)) / np.log(compute_p2(alpha / c))


# print(f'{compute_rho(1.0, 2.0, 3.0)=}, c=2, alpha=3')
# print(f'{compute_rho(2.0, 4.0, 6.0)=}, c=2, alpha=3')
# print(f'{compute_rho(1.5, 3.0, 4.5)=}, c=2, alpha=3')
# print(f'{compute_rho2(2,3)=}')


# search best alpha that minimizes rho for a given c
# In (R,c)-NN we need an algorithm with failure probability bounded by a constant f
# Setting L = N^rho ensures the failure probability of (R,c)-NN <= f = 1/3 + 1/e
#
# During construction space: O(dn + nL), time: O(N L k t_h)
# During query, time: O(L k t_h + 3L t_d)
# Minimizing rho will minimizes L
def find_best_alpha(c, alpha_lst=np.linspace(0, 30, 301)[1:]):
    if c > 20:
        print(f"warning {c=} too large, this method may not found best rho")
    rho = [compute_rho2(c, alpha) for alpha in alpha_lst]
    idx = np.argmin(rho)
    return alpha_lst[idx], rho[idx]


def compute_k(N, c, alpha):
    p2 = compute_p2(alpha / c)
    return np.ceil(np.log(N) / np.log(1 / p2))


def plot_p2():
    c_lst = [1.1, 1.5, 2, 2.5, 5, 10]
    p_lst = []
    for c in c_lst:
        alpha, rho = find_best_alpha(c)
        p2 = compute_p2(alpha / c)
        p_lst.append(1 / p2)

    plt.plot(c_lst, p_lst)
    plt.ylabel("1/p2")
    plt.xlabel("c")
    plt.title("1/p2 at best alpha for each c")
    plt.legend()
    plt.show()


# plot_p2()


###############################################################################
# 用 2-stable 分布构建 （r1,r2,p1,p2)-sensitive hashing function
# 用 k 次 and 扩大 p1^k 与 p2^k 的比值
# 用 L 个表保证选到一个比较近的邻居的概率足够大，失败的概率 f < 1/3 + 1/e
###############################################################################


class LSH_2_stable:
    def __init__(self, d, r, k, L, seed=None):
        """
        Generate a family of hashing functions
            h_{ij}(x) = \floor (a[i,j] \cdot x + b[i,j]) / r \floor
        So that g_i(x) = (h_{i0}(x), h_{i1}(x), ..., h_{ik}(x))

        Parameters
        ----------
        d : int
            dimension of data sample. Number of varianble in a training set.
        r : float
            bucket width
        k : int
            number of H in and construction for each g_i
        L : int
            number of hash tables (or construction)
        seed : TYPE, optional
            If not None pass to np.random.seed to fix random_state for reproduction of exact computation. The default is None.

        Returns
        -------
        None.

        """
        if seed is not None:
            np.random.seed(seed)

        self.a = np.random.normal(loc=0, scale=1, size=(L, k, d))
        self.b = np.random.uniform(low=0, high=1, size=(L, k))
        self.r = r
        self.d = d
        self.tables = [{} for i in range(L)]
        self.X = np.zeros((0, d))

    def compute_g(self, q, i=None):
        """
        Compute hash keys for sample q

        Parameters
        ----------
        q : np.array of shape (d,)
            a d-dimensional vector representing a data sample.

        Returns
        -------
        keys : np.array of (k,) or np.array of (L,k)
            when i is not None return keys = g_i(q) = (h_{i0}(q), h_{i1}(q), ..., h_{ik}(q))
            otherwise keys[i] = g_i(q) = (h_{i0}(q), h_{i1}(q), ..., h_{ik}(q))
        """
        if i is not None:
            # r[j] = \lfloor self.a[i,j] dot q + self.b[i,j] / self.r \rfloor
            return np.floor((np.matmul(self.a[i], q) + self.b[i]) / self.r)
        else:
            # r[i,j] = \lfloor self.a[i,j] dot q + self.b[i,j] / self.r \rfloor
            return np.floor((np.matmul(self.a, q) + self.b) / self.r)

    def add_all(self, X):
        """
        Append all samples to the end of self.X,
        for new x, add their index (as in self.X) to the bucket for g_i(x) in each table

        Parameters
        ----------
        X : np.array of shape (n, d)
            n samples to be added

        Returns
        -------
        None.

        """
        old_len = len(self.X)
        self.X = np.vstack((self.X, X))
        for idx, x in enumerate(X):
            new_idx = idx + old_len
            keys = self.compute_g(x)
            for i, key in enumerate(keys):
                key = tuple(key)
                lst = self.tables[i].get(key, [])
                lst.append(new_idx)
                self.tables[i][key] = lst

    def find_all(self, q, cR=None, max_items=np.infty):
        """
        Find all p in same bucket as q in all L tables, such that ||p - q||_2 < cR.
        Process table by table, until the number of items processed reaches max_items.

        Parameters
        ----------
        q : np.array of shape (d,)
            the data points to be queried.
        cR : float, optional
            Threshold of distance. If not None, only p with ||p - q||_2 <= cR is returned.
        max_items : int, optional
            Maximum number of items to be searched. Sariel Har-Peled et. al. 2012 set this to 3L. The default is np.infty.

        Returns
        -------
        results : [int]
            index of items found, empty list if no such item exists.
        """
        keys = self.compute_g(q)
        results = set()
        count = 0
        for i, key in enumerate(keys):
            key = tuple(key)
            idx_lst = self.tables[i].get(key, [])
            if cR is not None:
                P = self.X[idx_lst]  # P[i] = self.X[idx_lst[i]], for all i
                diff = P - q  # diff[i] = P[i] - q
                dist = np.linalg.norm(
                    diff, axis=1
                )  # dist[i] = || diff[i] ||_2 = || P[i] -q ||_2
                selected_idx = np.array(idx_lst)[dist < cR]
                results.update(selected_idx)
            else:
                results.update(idx_lst)
            count += len(idx_lst)
            if count >= max_items:
                break
        return list(results)


# @article{AlvarLSH2016,
# title = {Instance selection of linear complexity for big data},
# journal = {Knowledge-Based Systems},
# volume = {107},
# pages = {83-95},
# year = {2016},
# issn = {0950-7051},
# doi = {https://doi.org/10.1016/j.knosys.2016.05.056},
# url = {https://www.sciencedirect.com/science/article/pii/S0950705116301617},
# author = {Álvar Arnaiz-González and José-Francisco Díez-Pastor and Juan J. Rodríguez and César García-Osorio},
# keywords = {Nearest neighbor, Data reduction, Instance selection, Hashing, Big data},
# abstract = {Over recent decades, database sizes have grown considerably. Larger sizes present new challenges, because machine learning algorithms are not prepared to process such large volumes of information. Instance selection methods can alleviate this problem when the size of the data set is medium to large. However, even these methods face similar problems with very large-to-massive data sets. In this paper, two new algorithms with linear complexity for instance selection purposes are presented. Both algorithms use locality-sensitive hashing to find similarities between instances. While the complexity of conventional methods (usually quadratic, O(n2), or log-linear, O(nlogn)) means that they are unable to process large-sized data sets, the new proposal shows competitive results in terms of accuracy. Even more remarkably, it shortens execution time, as the proposal manages to reduce complexity and make it linear with respect to the data set size. The new proposal has been compared with some of the best known instance selection methods for testing and has also been evaluated on large data sets (up to a million instances).}
# }


def random_select_from_bucket(p_count, idx_lst_in_bucket):
    """
    Given a list of indices of sample in a bucket, randomly select one for each class
    if the class contains more than one sample.

    Parameters
    ----------
    p_count : int
        number of positive samples, indices that are smaller than p_count are positive samples
    idx_lst_in_bucket : list of int
        indices of samples.
    seed : int, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    selected_idx : TYPE
        DESCRIPTION.

    """
    p_idx_lst = [i for i in idx_lst_in_bucket if i < p_count]
    n_idx_lst = [i for i in idx_lst_in_bucket if i >= p_count]

    selected_idx = []
    if (
        len(p_idx_lst) > 1
    ):  # when other class has more instance, a class has 1 instance is likely to be noise
        rand_idx = np.random.choice(p_idx_lst)
        selected_idx.append(rand_idx)
    elif (
        len(p_idx_lst) == 1 and len(n_idx_lst) <= 1
    ):  # modified by zhuwb, when other class has 1 or less instance, it should not beconsidered as noise
        selected_idx.append(p_idx_lst[0])

    if len(n_idx_lst) > 1:
        rand_idx = np.random.choice(n_idx_lst)
        selected_idx.append(rand_idx)
    elif (
        len(n_idx_lst) == 1 and len(p_idx_lst) <= 1
    ):  # modified by zhuwb, when other class has 1 or less instance, it should not beconsidered as noise
        selected_idx.append(n_idx_lst[0])
    return selected_idx


def LSH_IS_F(ts, w=1.0, k=10, L=5, seed=1):
    d = ts.P.shape[1]
    lsh = LSH_2_stable(d, r=w, k=k, L=L, seed=seed)

    # A[i,j] --> [0,1], each col divid by range
    # TODO normalization using sklearn
    A = np.vstack((ts.P, ts.N))
    max_A = np.max(A, axis=0)
    min_A = np.min(A, axis=0)
    range_A = max_A - min_A
    zero_idx = range_A == 0
    range_A[zero_idx] = 1
    normalized_A = (A - min_A) / range_A

    lsh.add_all(normalized_A)
    processed = np.zeros((len(normalized_A)))
    selected_idx = []
    for i, q in enumerate(normalized_A):
        if processed[i] > 0:
            continue
        results = lsh.find_all(q)

        if len(results) > 0:
            sel_idx_per_bucket = random_select_from_bucket(len(ts.P), results)
            selected_idx.extend(sel_idx_per_bucket)

        # all neighbors in same bucket as
        processed[results] = 1
    return selected_idx


def LSH_IF_F_binary_search(ts, alpha=0.1, k=10, L=5, seed=1):
    w_min = 1e-6
    target = int((len(ts.P) + len(ts.N)) * alpha)
    # print(f"{target=}")

    # find a w_max, where selected > target+1
    w_max = 1.0
    MAX_ITER = 32
    for i in range(MAX_ITER):
        selected_idx = LSH_IS_F(ts=ts, w=w_max, k=k, L=L, seed=seed)
        # print(f"{w_max=}, {len(selected_idx)=}")
        if len(selected_idx) < target:
            break
        w_max *= 2
        if i == MAX_ITER - 1:
            warnings.warn(f"{MAX_ITER=} reached, {w_max=} may be too small.")

    if abs(len(selected_idx) - target) <= 1:
        return selected_idx, w_max

    # we are sure that w_min <= w* <= w_max
    for i in range(MAX_ITER):
        w_mid = (w_min + w_max) / 2
        selected_idx = LSH_IS_F(ts=ts, w=w_mid, k=k, L=L, seed=seed)
        # print(f"{w_mid=}, {len(selected_idx)=}, [{w_min}, {w_max}]")
        diff = len(selected_idx) - target
        if abs(diff) <= 1 or (w_max - w_min) < 1e-6:
            return selected_idx, w_mid

        if diff > 0:
            w_min = w_mid
        elif diff < 0:
            w_max = w_mid

        if i == MAX_ITER - 1:
            warnings.warn(f"{MAX_ITER=} reached, {w_mid=} may be too small")
            return selected_idx, w_mid


def lsh_plot(ts, selected_idx):
    if ts.P.shape[1] != 2:
        raise RuntimeError("the dataset is not 2D, it can not be visualized")

    if not isinstance(selected_idx, np.ndarray):
        selected_idx = np.array(selected_idx)

    plt.scatter(ts.P[:, 0], ts.P[:, 1], edgecolors="b", facecolors="none", s=150)
    plt.scatter(ts.N[:, 0], ts.N[:, 1], edgecolors="r", facecolors="none", s=150)

    A = np.vstack((ts.P, ts.N))

    selected_p_idx = selected_idx[selected_idx < len(ts.P)]
    selected_n_idx = selected_idx[selected_idx >= len(ts.P)]
    if len(selected_p_idx) > 0:
        select_p = A[selected_p_idx]
        plt.scatter(select_p[:, 0], select_p[:, 1], c="blueviolet", s=150)
    if len(selected_n_idx) > 0:
        select_n = A[selected_n_idx]
        plt.scatter(select_n[:, 0], select_n[:, 1], c="deeppink", s=150)


if __name__ == "__main__":
    # lsh = LSH_2_stable(d = 2, r = 1.0, k = 2, L=1, seed=0)
    # x = np.array([1,0])
    # print(f"{lsh.a = }, {lsh.b=} ")
    # keys = lsh.compute_g(x) # key[0,0,0] = 2, key[0,0,1] = 1
    # a0 = lsh.a[0,0]
    # a1 = lsh.a[0,1]
    # b0 = lsh.b[0,0]
    # b1 = lsh.b[0,1]
    # h0 = (a0[0] + b0)/1.0   # 2.18
    # h1 = (a1[0] + b1)/1.0   # 1.62
    # print(f"{h0 = }, {h1=}")

    N = 2000
    ts = ds.loadTwoGuasianNoise(noise=0.6, n_samples=[int(N / 2), int(N / 2)], seed=100)
    selected_idx, w_mid = LSH_IF_F_binary_search(ts, alpha=0.4, k=10, L=5, seed=1)

    # N = 2000
    # ts = ds.loadTwoGuasianNoise(noise=0.6,n_samples=[int(N/2),int(N/2)],seed = 100)
    # w_lst = [round(i,2) for i in np.arange(0.1,1.0,0.1)]
    # for w in w_lst:
    #     selected_idx = LSH_IS_F(ts, w=w, k=10, L=5, seed=1)
    #     lsh_plot(ts, selected_idx)

    #     plt.title(f"{w = }")
    #     plt.show()
    #     print(f"{w = }, selcted_ratio = {len(selected_idx)/(len(ts.P) + len(ts.N))}")
