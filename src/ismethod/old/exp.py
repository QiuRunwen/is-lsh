# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 20:44:15 2021
所有数据集每个算法都用 grid search 搜索一个最好的参数
筛选数据集的方法：
1. AC
2. AC + 2*diversity
3. OutDist
4. OutDist + 2*diversity
5. Freq
6. Freq + 2*diversity
7. Random Selection
8. Select by Voronoi
@author: dhlsf
"""

import random

import numpy as np
from scipy.spatial import distance
from sklearn import metrics

from . import dataset as ds
from . import proximity as pr
from . import selectTopK as dr
from .lsh_methods import LSH_IS_F, LSH_IF_F_binary_search
from .voronoi_explore import remove_noise_by_voronoi


def selsectByProximityAC(ts:ds.TrainSet, p_count, n_count, w, k=20, dist=distance.euclidean):
    return dr.selectTopKAsDS(
        ts=ts,
        p_count=p_count,
        n_count=n_count,
        p=pr.computeClosenessToMarginByAlternatingChain,
        k=k,
        w=w,
        dist=dist,
    )


def selectByProximityOutDist(ts:ds.TrainSet, p_count, n_count, w, k=20, dist=distance.euclidean):
    return dr.selectTopKAsDS(
        ts=ts,
        p_count=p_count,
        n_count=n_count,
        p=pr.computeClosenessToMarginByAvgOppKNNDist,
        k=k,
        w=w,
        dist=dist,
    )


def selectByProximityFrequencyInOppKNN(
    ts:ds.TrainSet, p_count, n_count, w, k=20, dist=distance.euclidean
):
    return dr.selectTopKAsDS(
        ts=ts,
        p_count=p_count,
        n_count=n_count,
        p=pr.computeClosenessToMarginByFreqeucyInOppKNN,
        k=k,
        w=w,
        dist=dist,
    )


def selectByProximityEntropy(ts:ds.TrainSet, p_count, n_count, w, k=20, dist=distance.euclidean):
    return dr.selectTopKAsDS(
        ts=ts,
        p_count=p_count,
        n_count=n_count,
        p=pr.computeClosenessToMarginByEntropy,
        k=k,
        w=w,
        dist=dist,
    )


# 1. select a subset by AC
def byAC(ts:ds.TrainSet, p_count, n_count, *args, **kwargs):
    return selsectByProximityAC(ts, p_count, n_count, w=0)


# 2. select a subset by AC + diversity
def byACDr(ts:ds.TrainSet, p_count, n_count, *args, **kwargs):
    return selsectByProximityAC(ts, p_count, n_count, w=1)


# 3. select a subset by OutDist
def byOutdist(ts:ds.TrainSet, p_count, n_count, *args, **kwargs):
    return selectByProximityOutDist(ts, p_count, n_count, w=0)


# 4. select a subset by OutDist + diversity
def byOutdistDr(ts:ds.TrainSet, p_count, n_count, *args, **kwargs):
    return selectByProximityOutDist(ts, p_count, n_count, w=1)


# 5. select a subset by Freq
def byFreq(ts:ds.TrainSet, p_count, n_count, *args, **kwargs):
    return selectByProximityFrequencyInOppKNN(ts, p_count, n_count, w=0)


# 6. select a subset by Freq + diversity
def byFreqDr(ts:ds.TrainSet, p_count, n_count, *args, **kwargs):
    return selectByProximityFrequencyInOppKNN(ts, p_count, n_count, w=1)


def sample_idx(n, m, rand):  # randomly select m from 0,1,...,(n-1)
    idx_lst = [i for i in range(n)]
    rand.shuffle(idx_lst)
    return idx_lst[:m]


# 7. by random sampling
def bySamplingSeed4(ts:ds.TrainSet, p_count, n_count, seed=4, *args, **kwargs):
    """
    随机筛选
    """
    rand = random.Random(seed)
    p_count = min(p_count, len(ts.P))
    n_count = min(n_count, len(ts.N))

    selected_P_idx = sample_idx(len(ts.P), p_count, rand)
    selected_N_idx = sample_idx(len(ts.N), n_count, rand)

    P = ts.P[selected_P_idx]
    PW = ts.PW[selected_P_idx]
    N = ts.N[selected_N_idx]
    NW = ts.NW[selected_N_idx]
    return ds.TrainSet(P, PW, N, NW, ts.dsDir)


# 8. select a subset by Voronoi no removal
def byVoronoiNoFilter(ts:ds.TrainSet, p_count, n_count, *args, **kwargs):
    ds_v = remove_noise_by_voronoi(ts, min_dist_to_boundary=-2)
    return ds_v.getSubset(p_count, n_count)


# 9. select a subset by Voronoi with filter 0
def byVoronoiFilter0(ts:ds.TrainSet, p_count, n_count, *args, **kwargs):
    ds_v = remove_noise_by_voronoi(ts, min_dist_to_boundary=0)
    return ds_v.getSubset(p_count, n_count)


# 10. select a subset by Voronoi with filter 0.2
def byVoronoiFilter02(ts:ds.TrainSet, p_count, n_count, *args, **kwargs):
    ds_v = remove_noise_by_voronoi(ts, min_dist_to_boundary=0.2)
    return ds_v.getSubset(p_count, n_count)


# 11. select a subset by Voronoi with filter 0.5
def byVoronoiFilter05(ts:ds.TrainSet, p_count, n_count, *args, **kwargs):
    ds_v = remove_noise_by_voronoi(ts, min_dist_to_boundary=0.5)
    return ds_v.getSubset(p_count, n_count)


# 12. select a subset by Entropy k=10
def byEntropy10(ts:ds.TrainSet, p_count, n_count, *args, **kwargs):
    return selectByProximityEntropy(ts, p_count, n_count, w=0, k=10)


# 13. select a subset by Entropy k=20
def byEntropy20(ts:ds.TrainSet, p_count, n_count, *args, **kwargs):
    return selectByProximityEntropy(ts, p_count, n_count, w=0, k=20)


def byLSH_IS_F_w(ts:ds.TrainSet, p_count, n_count, w_alpha=0.1, *args, **kwargs):
    selected_idx = LSH_IS_F(ts, w=w_alpha)
    X, y = ts.toXY()
    X_selected_sub = X[selected_idx]
    y_selected_sub = y[selected_idx]
    return ds.TrainSet.gen(
        X_selected_sub, y_selected_sub
    )


def byLSH_IS_F_bs(ts:ds.TrainSet, p_count, n_count, w_alpha=0.1, *args, **kwargs):
    selected_idx, w = LSH_IF_F_binary_search(ts, alpha=w_alpha)
    X, y = ts.toXY()
    if not isinstance(selected_idx, np.ndarray):
        selected_idx = np.array(selected_idx)
    selected_idx = selected_idx.astype(int)
    X_selected_sub = X[selected_idx]
    y_selected_sub = y[selected_idx]
    return ds.TrainSet.gen(
        X_selected_sub, y_selected_sub
    )


allStrategyList = [
    byAC,
    byACDr,
    byOutdist,
    byOutdistDr,
    byFreq,
    byFreqDr,
    bySamplingSeed4,
    byVoronoiNoFilter,
    byVoronoiFilter0,
    byVoronoiFilter02,
    byVoronoiFilter05,
    byEntropy10,
    byEntropy20,
    byLSH_IS_F_bs,
]


if __name__ == "__main__":
    exp_strategy_list = [
        # byLSH_ours, byLSH_ours_noUseW, byLSH_ours_useOpArg,
        # LSH_ours, LSH_ours_OpArg_R01,LSH_ours_OpArg_R02,LSH_ours_OpArg_R04,LSH_ours_OpArg_R06,LSH_ours_OpArg_R08,
        # bySamplingSeed4,
        byLSH_IS_F_bs,
        byVoronoiFilter02,
        byAC,
        byFreq,
        byEntropy20,
    ]
