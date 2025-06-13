# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 20:34:52 2020

@author: dhlsf
"""

import os

import numpy as np
from scipy.spatial import distance

from . import dataset as ds
from . import proximity


def findNeighbors(P, X, k, dist=distance.euclidean):
    """
    给P中的每个点在X中找最近的k个邻居

    Parameters
    ----------
    P : numpy.array
        P[i]表示P中的第i个正样本/负样本数据点
    X : numpy.array
        X[i]表示数据集中的第i个样本点.
    k : int
        邻居的个数.
    dist : function, optional
        计算两个点之间的距离的函数，默认为欧式距离

    Returns
    -------
    nc : list
        每个点的k个邻居的index以及对应的距离.
        nc[i] = [(i_0,d_0),(i_1,d_1),(i_2,d_2),...]
        i_0为P中第i个点的第一个邻居，d_0为对应的距离，以此类推...
    """
    n = len(P)
    nc = []

    import time

    startTime = time.time()
    nextPrintTime = 0
    for i in range(n):
        nc.append(proximity.KNN(P[i], X, k, dist=dist))

        elapsedTime = time.time() - startTime
        if elapsedTime >= nextPrintTime:
            print(
                "findNeighors: {0}/{1}, elapsed time: {2} s".format(i, n, elapsedTime)
            )
            nextPrintTime += 30
    return nc


def normalDataIndex(P, N, k, dist=distance.euclidean):
    """
    分别计算正样本和负样本中每个点的是正常数据（非噪声）的指标

    Parameters
    ----------
    P : numpy.array
        P[i]表示第i个正样本点.
    N : numpy.array
        N[i]表示第i个负样本点.
    k : int
        对噪音点进行度量的时候，每个点选择的邻居个数.
    dist : function, optional
        计算两个点之间的距离的函数，默认为欧式距离
    Raises
    ------
    RuntimeError
        对噪音点进行度量的时候，如果每个点选择的邻居个数过小(小于0)或者过大(大于所有样本点的个数)提示RuntimeError错误.

    Returns
    -------
    PW : float array
        PW[i] = p_i / maxNC * p_i / k
        p_i is number of positive neighobrs for P[i], maxNC is max p_i for all i in P
        PW[i]是对正样本中每个点噪音程度的度量，是[0,1]之间的数，越小则 P[i] 越有可能是噪音, 越大越是正常数据


    NW : float array
        NW[i] = n_i / maxNC * n_i / k
        n_i is number of negative neighbors for N[i], maxNC is max n_i for all i in N
        NW[i]是对负样本中每个点噪音程度的度量，是[0,1]之间的数，越小则 N[i] 越有可能是噪音, 越大越是正常数据
    """
    plen = len(P)
    nlen = len(N)

    if k < 0 or k >= (plen + nlen):
        raise RuntimeError(
            "k = {0} is not a positive int less than {1}".format(k, plen + nlen)
        )

    X = np.vstack((P, N))

    nc = findNeighbors(P, X, k, dist=dist)
    PW = np.zeros(plen)  # PW[i]: number of positive neighbors
    for i in range(plen):
        for idx, d in nc[i]:
            if idx < plen:
                PW[i] += 1

    nc = findNeighbors(N, X, k, dist=dist)
    NW = np.zeros(nlen)  # NW[i]: number of negative neighbors for i in N
    for i in range(nlen):
        for idx, d in nc[i]:
            if idx >= plen:
                NW[i] += 1

    maxNC = max(PW)
    if maxNC > 0:
        for i in range(plen):
            PW[i] = PW[i] / maxNC * PW[i] / k  # PW[i]: 对正样本中每个点噪音程度的衡量

    maxNC = max(NW)
    if maxNC > 0:
        for i in range(nlen):
            NW[i] = NW[i] / maxNC * NW[i] / k  # NW[i]： 对负样本中每个点噪音程度的衡量

    return PW, NW


def addNormalDataIndex(trainSet, k=20, dist=distance.euclidean):
    """

    给一个数据集加上数据正常程度衡量，计算PW,NW，并且根据 dsDir 是否有 nr-<dist>-k/PW.npy 决定是load还是计算并save

    Parameters
    ----------
    trainSet : class
        需要去噪音的数据集.
    k : int
        对噪音点进行度量的时候，每个点选择的邻居个数. The default is 20.
    dist : function, optional
        计算距离的函数. The default is distance.euclidean.

    Returns
    -------
    TYPE
        DESCRIPTION.
    """

    # dsDir = os.path.join(trainSet.dsDir, "nr-" + dist.__name__ + "-" + str(k))

    # filePW = os.path.join(dsDir, "PW.npy")
    # fileNW = os.path.join(dsDir, "NW.npy")

    # if os.path.exists(filePW):
    #     PW = np.load(filePW)
    #     NW = np.load(fileNW)

    # else:
    #     import time

    #     start = time.time()
    #     PW, NW = normalDataIndex(trainSet.P, trainSet.N, k=k, dist=dist)
    #     runTime = time.time() - start
        # if not os.path.exists(dsDir):
        #     os.makedirs(dsDir)

        # np.save(filePW, PW)
        # np.save(fileNW, NW)
        # np.save(os.path.join(dsDir, "time.npy"), runTime)
    PW, NW = normalDataIndex(trainSet.P, trainSet.N, k=k, dist=dist)
    dsDir = trainSet.dsDir
    return ds.TrainSet(trainSet.P, PW, trainSet.N, NW, dsDir)


def removeNoiseForList(ts, pnRemoveRatioList, k=20, dist=distance.euclidean):
    """
    pnRemoveRatioList = [(pr0, nr0), (pr1, nr1), ...]
    为数据集ts的正负类样本分别去掉 pri 和 nri 比例的数据点构成一个新数据集 ds[i]
    ds[i] 的 PW 与 NW 保留数据点在 ts 中的对应 PW 与 NW

    Parameters
    ----------
    ts : class
        需要去噪音的数据集.
    pnRemoveRatioList : list of pair of floats, optional
        [(pr1,nr1),(pr2,nr2),...]
        pri: 是第i次 正样本中去除的噪音比例.
        nri: 是第i次 负样本中去除的噪音比例.
    removeNRatio : int, optional
        负样本中去除噪音的比例. The default is 0.05.
    k : int, optional
        对噪音点进行度量的时候，每个点选择的邻居个数. . The default is 20.
    dist : function, optional
        计算距离的函数. The default is distance.euclidean.

    Returns
    -------
    class
        ds[i] 是 ts 按照 pnRemoveRatioList[i] = (pri, nri)
        正样本去除 pri，负样本去除 nri 比例噪音后的数据集.
    """
    dsWithNormalIdx = addNormalDataIndex(ts, k, dist)

    result = []
    for removalPRatio, removeNRatio in pnRemoveRatioList:
        if isinstance(ts, ds.TrainSetWithGT):
            newDS = dsWithNormalIdx.getSubsetByRatio(
                1 - removalPRatio, 1 - removeNRatio
            )
        else:
            newDS, pIdx, nIdx = dsWithNormalIdx.getSubsetByRatio(
                1 - removalPRatio, 1 - removeNRatio, returnIdx=True
            )
            newDS.PW = ts.PW[pIdx]
            newDS.NW = ts.NW[nIdx]
            result.append(newDS)
            continue

        if isinstance(ts, ds.TrainSet2Guasian):
            result.append(ds.TrainSet2Guasian(newDS.dsDir, newDS.P, newDS.N))
            continue

        if isinstance(ts, ds.TrainSet2Circle):
            result.append(ds.TrainSet2Circle(newDS.dsDir, newDS.P, newDS.N, ts.R))
            continue

        if isinstance(ts, ds.TrainSetWithGT):
            raise RuntimeError(
                "Please implement for data set type: {0}".format(type(ts))
            )

    return result


def removeNoise(ts, removalPRatio=0, removeNRatio=0.05, k=20, dist=distance.euclidean):
    return removeNoiseForList(ts, [(removalPRatio, removeNRatio)], k=k, dist=dist)[0]


import warnings


def removeNoiseForCircle(ts, removalPRatio=0, removeNRatio=0.05, k=20):
    """
    为数据集ts分别移在正负类样本中去掉removePRatio和removeNRatio比例的数据点
    返回的数据集 PW 与 NW 是到决策边界的距离的负数

    Parameters
    ----------
    ts : TYPE
        DESCRIPTION.
    removalPRatio : TYPE, optional
        DESCRIPTION. The default is 0.
    removeNRatio : TYPE, optional
        DESCRIPTION. The default is 0.05.

    Raises
    ------
    RuntimeError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    warnings.warn(
        "Warning: please replace removeNoiseForCircle by removeNoise",
        DeprecationWarning,
    )

    return removeNoise(ts, removalPRatio, removeNRatio, k=k)


def removeNoiseFor2Guasian(ts, removalPRatio=0, removeNRatio=0.05):
    warnings.warn(
        "Warning: please replace removeNoiseForCircle by removeNoise",
        DeprecationWarning,
    )
    if not isinstance(ts, ds.TrainSet2Guasian):
        raise RuntimeError("only support TrainSet2Guasian")

    print("Warning: please replace removeNoiseFor2Guasian by removeNoise")

    return removeNoise(ts, removalPRatio, removeNRatio)


if __name__ == "__main__":
    # # Two Circle数据集
    # a = ds.loadTwoCircleNoise()
    # na = addNormalDataIndex(a)
    # na.showWithTitle("normal data index")

    # b = na.getSubsetByRatio(0.9, 0.9)
    # b.showWithTitle("90% data")

    # b = na.getSubsetByRatio(0.8, 0.8)
    # b.showWithTitle("80% data")

    # Two Guassian数据集
    a = ds.loadTwoGuasianNoise()
    a.showWithTitle("Two Guassian Data")

    na01 = removeNoiseFor2Guasian(a, 0.1, 0.1)
    na01.showWithTitle("90% data")

    na = removeNoiseFor2Guasian(a, 0.2, 0.2)
    na.showWithTitle("80% data")

    from playsound import playsound

    playsound("../Canon.mp3")
