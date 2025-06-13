# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 11:26:18 2021
@author: dhlsf
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d

from . import dataset as ds

plt.style.use("ggplot")


def findAdjacentCells(v):
    """
    给定一个Voronoi图，找到每个 edge 对应的 region
    """
    regions = v.regions
    edge2RegionLst = {}
    for idx, region in enumerate(regions):
        for i in range(len(region)):
            idx1 = region[i]
            idx2 = region[(i + 1) % len(region)]
            if idx1 == idx2:
                raise RuntimeError
            minIdx = min(idx1, idx2)
            maxIdx = max(idx1, idx2)
            if minIdx == maxIdx:
                raise RuntimeError
            edge = (minIdx, maxIdx)
            if edge not in edge2RegionLst:
                edge2RegionLst[edge] = []
            edge2RegionLst[edge].append(idx)
    return edge2RegionLst


def regionToPointIdx(v):
    """
    给定一个Voronoi图，找 region 与 数据点 Idx 的对应关系
    """
    point2region = v.point_region
    point2regionDict = {}
    for i in range(len(point2region)):
        point2regionDict[i] = point2region[i]

    region2PointDict = {}
    for pointIdx, regionIdx in point2regionDict.items():
        region2PointDict[regionIdx] = pointIdx
    return region2PointDict


def getAdjacentPoints(X, y):
    v = Voronoi(X)
    edge2RegionLst = findAdjacentCells(v)
    region2PointDict = regionToPointIdx(v)

    adjacentPointPair = []
    for edge, regionLst in edge2RegionLst.items():
        if len(regionLst) != 2:
            print(f"{regionLst=}")
        r0 = regionLst[0]
        r1 = regionLst[1]
        p0 = region2PointDict[r0]
        p1 = region2PointDict[r1]
        adjacentPointPair.append((p0, p1))
    return adjacentPointPair


def point2Neighbor_2D(X, y, plot=False):
    adjacentPointPair = getAdjacentPoints(X, plot)
    point2Neighbor = {}
    for i in range(len(adjacentPointPair)):
        p0Idx = adjacentPointPair[i][0]
        p1Idx = adjacentPointPair[i][1]

        if p0Idx not in point2Neighbor.keys():
            point2Neighbor[p0Idx] = []

        if p1Idx not in point2Neighbor.keys():
            point2Neighbor[p1Idx] = []

        point2Neighbor[p0Idx].append(p1Idx)
        point2Neighbor[p1Idx].append(p0Idx)

    return point2Neighbor


#################### D > 2 #######################


def neighbor_by_ray(D, n, r):
    """
    以 X[i] 为起点 r 为方向的射线，如果射中一个邻居 X[j]， 返回 j 否则 -1
    Return j if a ray starting from X[i] in direction r hits Voronoi neighobr X[j],
    -1 otherwise.

    Parameters
    ----------
    D : 2D numpy array of shape (N,p)
        D[j] = X[j] - X[i] is the i-th point, p-dimensional vector
    n : 1D numpy array of shape (N,)
        n[j] = || D[j] ||^2
    i : int
        DESCRIPTION.
    r : 1D numpy array of shape (p,)
        Direction of ray

    Returns
    -------
    index of Voronoiy neighbor of X[i], -1 if no such neighbor exists

    """
    t = np.matmul(D, r)  # t_j = <D[j], r>

    # Let A be the intersection of the ray and the bisector of the line segment X[i],X[j]
    # The distance between X[i] and A is proportional to
    #   q(x_i, r, x_j) = n_j / t_j * (2 / ||r||^2)
    # X[j] is a potential neighbor only if r form a acute angle with line X[i], X[j]
    # *) When r is parpendicular to bisector (including X[i] == X[j]), there is
    #    no intersection (intersect at infinity) x_j is not neighbor
    # neighbor j
    #   j = arg min_{j, t_j > 0} (n_j/t_j)

    # Slow version
    # minDis = np.Infinity
    # minIdx= -1
    # for j in range(len(X)):
    #     if t[j] > 0:
    #         dis = n[j] / t[j]
    #         if dis < minDis:
    #             minDis = dis
    #             minIdx = j
    # return minIdx

    # Vectorized code
    candidate_flag = t > 0
    if np.sum(candidate_flag) > 0:
        candidate_idx = np.array(range(len(D)))[candidate_flag]
        neighbor_idx_in_candidate = np.argmin(n[candidate_flag] / t[candidate_flag])
        neighbor_idx = candidate_idx[neighbor_idx_in_candidate]
        return neighbor_idx
    return -1


def sample_points_on_surface_of_unit_ball(n, d):
    """
    Uniformly sample n points from the surface of a d-dimensional unit ball.
    从 d-维单位球的表面均匀的采样 n 个点
    http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/

    Parameters
    ----------
    d : int
        Dimensional of unit ball.
    n : int
        number of points to sample.

    Returns
    -------
    TYPE
        2D numpy array of shape (n,d)

    """
    r = np.zeros((n, d))
    for i in range(n):
        u = np.random.normal(
            0, 1, d
        )  # an array of d normally distributed random variables
        r[i] = u / np.linalg.norm(u)  # np.sum(u**2) ** (0.5)
    return r


import time


def point2Neighbor(X, sampleCount=360, seed=3):
    np.random.seed(seed)
    rays = sample_points_on_surface_of_unit_ball(sampleCount, X.shape[1])
    start_time = time.time()
    p2n = {}
    for i in range(len(X)):
        # print(f"{i = }, time={time.time()-start_time:.2f} s")
        D = X - X[i]  # D[j] = X[j] - X[i]
        n = np.sum(D**2, axis=1)  # n_j = sum_i (D[j,i] * D[j,i]) = || D[j] ||^2
        neighborSet = set()
        for r in rays:
            minIdx = neighbor_by_ray(D, n, r)
            if minIdx >= 0:
                neighborSet.add(minIdx)
        p2n[i] = list(neighborSet)
    return p2n


def list_equal(L1, L2):
    return len(L1) == len(L2) and set(L1) == set(L2)


def graph_equal(p2n, exp_p2n):
    if len(p2n) != len(exp_p2n):
        return False
    for p in p2n:
        L1 = p2n[p]
        L2 = exp_p2n[p]
        if not list_equal(L1, L2):
            return False
    return True


#################### General Code ####################


def overlapIndex(X, y, point2Neighbor):
    """
    Compute pdoc(x_i, y_i) for all (x_i, y_i) and their avereage as overlap index oi

    Parameters
    ----------
    X : np.ndarry with shape (N,l)
        x[i] is i-th instance with l dimensons
    y : np.ndarray with shape （N,)
        y[i] is 1 or -1
    point2Neighbor : dict{int: list(int)}
        point2Neighbor[i] is list of index of neighbors of X[i]

    Returns
    -------
    avg_overlap : float
        The overlap index oi, average pdoc of all instances
    dist_to_boundary : np.nadarray of shape (N,)
        dist_to_boundary[i] is pdoc(x_i, y_i)

    """
    prob_density_diff = np.zeros((len(X),))  # 点 i 附近正样本概率密度与负样本概率密度的差
    for pIdx, neighborLst in point2Neighbor.items():
        if len(neighborLst) > 0:
            prob_density_diff[pIdx] = np.mean(y[neighborLst])

    dist_to_boundary = (
        prob_density_diff * y
    )  # [-1, 1] 之间， 负数表示在决策面错误的一侧，正数表示在决策面正确的一侧，绝对数越小距离决策面越近

    # if distPlot:
    #     plt.hist(labelIndex,bins=20)
    #     plt.show()

    avg_overlap = np.mean(dist_to_boundary)

    return avg_overlap, dist_to_boundary


def yTransform(y):
    """
    y should be numpy array
    """
    y_copy = y.copy()
    y_copy = np.where(y == 0, -1, y)
    return y_copy


# PDOC-V algorithm in paper
def remove_noise_by_voronoi(
    ts, min_dist_to_boundary=0.2, p2n_method=point2Neighbor, sampleCount=360
):
    """
    快速构建 voronoi 邻居，用邻居估算每个点附近的正负样本密度差。
    把数据集 ts 中 dist_to_boundary <= min_dist_to_boundary 删除，创建一个子集 nts
    nts.avg_overlap 会计算 ts 整个数据集的 overlap_index

    Parameters
    ----------
    ts : TYPE
        DESCRIPTION.
    min_dist_to_boundary : TYPE, optional
        DESCRIPTION. The default is 0.2.
    p2n_method : TYPE, optional
        DESCRIPTION. The default is point2NeighborFast.
    sampleCount : TYPE, optional
        DESCRIPTION. The default is 360.

    Returns
    -------
    ts : TYPE
        DESCRIPTION.

    """
    X, y = ts.toXY()

    y = yTransform(y)  # 0 -> -1

    # load or save point_to_neighor
    # p2n_save_path = os.path.join(
    #     ts.dsDir, f"p2n_{p2n_method.__name__}_sampleCount={sampleCount}.pkl"
    # )
    # try:
    #     p2n = pickle.load(open(p2n_save_path, "rb"))
    # except (OSError, IOError, FileNotFoundError):
    #     if X.shape[1] > 2:
    #         p2n = point2Neighbor(X, sampleCount)  # shooting rays to find neighbors
    #     else:  # 维度小于等于2时掉包精确计算
    #         p2n = point2Neighbor_2D(X, y)

    #     pickle.dump(p2n, open(p2n_save_path, "wb"))
    if X.shape[1] > 2:
        p2n = point2Neighbor(X, sampleCount)  # shooting rays to find neighbors
    else:  # 维度小于等于2时掉包精确计算
        p2n = point2Neighbor_2D(X, y)

    avg_overlap, dist_to_boundary = overlapIndex(X, y, p2n)

    selected_idx = dist_to_boundary > min_dist_to_boundary
    X_selected = X[selected_idx]
    y_selected = y[selected_idx]
    d_selected = dist_to_boundary[selected_idx]

    P = X_selected[y_selected > 0]
    PW = -d_selected[y_selected > 0]
    N = X_selected[y_selected <= 0]
    NW = -d_selected[y_selected <= 0]

    nts = ds.TrainSet(P, PW, N, NW, ts.dsDir)
    nts.avg_overlap = avg_overlap

    return nts


if __name__ == "__main__":
    # example 1
    X = np.array(
        [
            [1, 3.8],
            [1.2, 4.5],
            [2.5, 3.9],
            [1.8, 3.8],
            [1.5, 5.5],
            [2, 5],
            [1.9, 4.5],
            [2.1, 3.5],
            [2.8, 4],
            [3, 3.5],
            [1.5, 4.8],
            [1.5, 4.2],
        ]
    )
    y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    v = Voronoi(X)
    voronoi_plot_2d(v, show_vertices=False)
    plt.scatter(X[:, 0], X[:, 1], s=200)
    for i in range(len(X)):
        plt.annotate(str(i), (X[i][0], X[i][1] + 0.05))
    p2n = point2Neighbor(X, sampleCount=360)
    p2n_2D = point2Neighbor_2D(X, y)
    exp_p2n = {
        0: [3, 1, 11, 7],
        1: [0, 10, 11, 4],
        2: [3, 6, 7, 8, 9],
        3: [0, 2, 6, 7, 11],
        4: [1, 10, 5],
        5: [8, 10, 4, 6],
        6: [2, 3, 5, 8, 10, 11],
        7: [0, 9, 2, 3],
        8: [9, 2, 5, 6],
        9: [8, 2, 7],
        10: [1, 4, 5, 6, 11],
        11: [0, 1, 3, 6, 10],
    }
    print(f"{p2n=}, {graph_equal(p2n,exp_p2n)=}")
    print("-----------------------")
    print(f"{p2n_2D = }, {graph_equal(p2n_2D,exp_p2n)=}")

    # # example 2
    # from sklearn.datasets import load_breast_cancer
    # import time

    # X = load_breast_cancer().data
    # y = load_breast_cancer().target
    # startTime = time.time()

    # dsDir = "../result/test-0401"
    # os.makedirs(dsDir,exist_ok=True)
    # ts = ds.TrainSet.gen(X,y,dsDir)

    # nts = remove_noise_by_voronoi(ts,min_dist_to_boundary=0.2,p2n_method=point2Neighbor,sampleCount=360)
    # sub = nts.getSubsetByRatio(pratio=0.2,nratio=0.2)
