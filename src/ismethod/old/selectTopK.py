# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 22:05:38 2020

@author: dhlsf

贪心方法：
step 1: 找一个离边界最近的点
step 2: 找一个与1不同的最近点
step 3: 找一个与1和2不同的最近点
...
用S表示已经选中的i个点，第i+1个点需要满足的条件：
a) 离边界近
b) 与S中各点都不同--->与S中最近点距离比较大
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance

from . import dataset as ds
from . import proximity, util
from .noiseRemoval import removeNoiseFor2Guasian, removeNoiseForCircle
from .proximity import addProximity, proximityNames


# =============================================================================
# 根据proximity + w*diversity的实现
# =============================================================================
def minDist(p, S, dist=distance.euclidean):
    """
    找出p与S中各个点的最小距离

    """
    if len(S) == 0:
        return 0

    minDis = math.inf

    for i in range(len(S)):
        d = dist(p, S[i])
        if d < minDis:
            minDis = d

    return minDis


def selectTopk(origP, origPW, k):
    """
    通过贪心的算法在origP中找出top K个点，使得这 K 个点离边界近，同时与S中各点不相同

    Parameters
    ----------
    origP : numpy.array
        origP[i]: 正样本点中第 i 个正样本点

    origPW : numpy.array
        origPW[i]: 正样本点中第 i 个正样本点离边界的proximity

    k : int
        需要从正样本中挑选符合要求(离边界近，同时又与S中各点不相同)的点的个数.

    Returns
    -------
    S : numpy.array
        通过贪心的算法在origP中找出的离边界近，同时与S中各点不相同top K个点.

    """
    P = np.copy(origP)
    PW = np.copy(origPW)

    S = []
    sPW = np.zeros(k)
    Plen = len(P)  # P[0:plen]表示剩余点， PW[0:plen]表示剩余点的priority

    for i in range(k):
        # 在P中找离边界近的与S不同的点
        maxV = -math.inf
        maxp = -1  # 下标

        for p in range(Plen):
            v = minDist(P[p], S) + PW[p]
            if v > maxV:
                maxV = v
                maxp = p

        #        print("++++ i: {0}; maxp: {1}; maxV: {2}".format(i, maxp, maxV))

        # add p to S
        S.append(np.copy(P[maxp]))  # S.append(P[maxp]) 只是复制了地址，迟一点这个位置如果内容变了，S也被不小心变了
        sPW[i] = PW[maxp]

        # remove p from P
        Plen = Plen - 1
        P[maxp] = P[Plen]
        PW[maxp] = PW[Plen]

    return np.array(S), sPW


def selectTopkFast(origP, origPW, k, mdWeight=1.0, dist=distance.euclidean):
    """
    通过贪心的算法在origP中找出top K个点，使得这 K 个点离边界近，同时与S中各点不相同

    Parameters
    ----------
    origP : numpy.array
        origP[i]: 正样本点中第 i 个正样本点

    origPW : numpy.array
        origPW[i]: 正样本点中第 i 个正样本点离边界的proximity

    k : int
        需要从正样本中挑选符合要求(离边界近，同时又与S中各点不相同)的点的个数.

    mdWeight : TYPE, optional
        diversity需要的权重. The default is 1.0.
        mdWeight = w*weightP
    dist : TYPE, optional
        DESCRIPTION. The default is distance.euclidean.

    Returns
    -------
    TYPE
        DESCRIPTION.
    sPW : TYPE
        sPW[i] 记录 S[i] 对应的 proximity + minDist.
    sI : TYPE
        DESCRIPTION.

    """
    plen = len(origP)
    P = np.copy(origP)
    PW = np.copy(origPW)
    idx = [i for i in range(plen)]  # idx[i] 是 P[i] 在origP 中的下标

    # # debug
    # print("========== begin =============")
    # print("P: ",P)
    # print("PW: ",PW)
    # print("idx: ",idx)
    # print("plen: ",plen)

    # 找第一个点
    maxp = np.argmax(PW)

    # add p to S
    point = np.copy(P[maxp])
    # S = [np.copy(point)]   # S 记录top K个点; S = [point] 如果一个点超过1维，是用数组保存，point是数组的地址，后来P[maxp] = P[plen] 会替换掉它
    sPW = np.zeros(k)  # sPW[i] 记录 S[i] 对应的 proximity + minDist
    sPW[0] = PW[maxp]
    sI = [maxp]  # 记录选择的 k 个点的原始下标, 一开始 idx[maxp] 就是 maxp

    # 删除第一个点
    plen -= 1
    P[maxp] = P[plen]
    PW[maxp] = PW[plen]
    idx[maxp] = idx[plen]

    # md[i] 是 P[i] 到 S 的最小距离， S 目前只有一个点
    md = np.zeros(plen)
    for j in range(plen):
        md[j] = dist(point, P[j])

    import time

    nextPrintTime = time.time()

    for i in range(1, k):
        if time.time() > nextPrintTime:
            nextPrintTime += 30
            print("select {0:6.0f}/{1:6.0f}".format(i, k))

        # 找到第 i 个点
        maxp = np.argmax(mdWeight * md[0:plen] + PW[0:plen])

        # add p to S
        point = np.copy(P[maxp])
        # S.append(np.copy(point)) # 错误 S.append(point)
        sPW[i] = PW[maxp]
        sI.append(idx[maxp])

        # remove p from P
        plen -= 1
        P[maxp] = P[plen]
        PW[maxp] = PW[plen]
        md[maxp] = md[plen]
        idx[maxp] = idx[plen]

        # 当 S' = S + {point} 时 md'[i] 要用新加入的点 point 更新
        for j in range(plen):
            md[j] = min(md[j], dist(point, P[j]))

        # # debug
        # print("------ loop ---------, i: ",i, " maxp: ",maxp)
        # print("P: ",P[0:plen])
        # print("PW: ",PW[0:plen])
        # print("idx: ",idx[0:plen])
        # print("md: ",md[0:plen])
        # print("plen: ",plen)
        # print("sI: ",sI)

    return origP[sI], sPW, sI


def selectTopKAsDS(ts, p_count, n_count, p, k, w, dist=distance.euclidean):
    """
    根据 Proximity (p, k) + w * Diversity 从 ts 的正负样本中各挑选 pratio, nratio 数据
    构成一个新的数据集

    Parameters
    ----------
    ts : 一个数据集
        DESCRIPTION.
    p_count : int
        挑选正样本的个数.
    n_count : int
        挑选负样本的个数.
    p : 计算 Proximity 的方法
        DESCRIPTION.
    k : Proximity 中的参数 k
        DESCRIPTION.
    w : Diversity 相对与 Proxmity 的权重
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    p_count = min(p_count, len(ts.P))
    n_count = min(n_count, len(ts.N))

    nds = addProximity(ts, k=k, clossnessCalc=p, dist=dist)
    if w == 0:
        return nds.getSubset(p_count, n_count)
    weightP, weightN = computeMDWeightForProximity(ts, p)

    # filePrefix = "p={0}_k={1}_w={2}".format(proximity.proximityNames[p], k, w)
    # sPfile = os.path.join(nds.dsDir, filePrefix + "_sP.npy")
    # sPWfile = os.path.join(nds.dsDir, filePrefix + "_sPW.npy")
    # sNfile = os.path.join(nds.dsDir, filePrefix + "_sN.npy")
    # sNWfile = os.path.join(nds.dsDir, filePrefix + "_sNW.npy")
    # if os.path.exists(sPfile):
    #     print("load from file: " + sPfile)
    #     sP = np.load(sPfile, allow_pickle=True)
    #     sPW = np.load(sPWfile, allow_pickle=True)
    #     sN = np.load(sNfile, allow_pickle=True)
    #     sNW = np.load(sNWfile, allow_pickle=True)
    #     if p_count != len(sP):
    #         raise RuntimeError(
    #             f"{p_count} != {len(sP)}, number of positive samples changed since last save. Check if train data is changed. Possibly by different random seed during generation or sampling. Quick fix: choose a new dsDir"
    #         )
    #     if n_count != len(sN):
    #         raise RuntimeError(
    #             f"{n_count} != {len(sN)}, number of negative samples changed since last save. Check if train data is changed. Possibly by different random seed during generation or sampling. Quick fix: choose a new dsDir"
    #         )
    # else:
    #     import time

        # start = time.time()
        # sP, sPW, sIP = selectTopkFast(nds.P, nds.PW, p_count, mdWeight=weightP * w)
        # sN, sNW, sIN = selectTopkFast(nds.N, nds.NW, n_count, mdWeight=weightN * w)
        # runTime = time.time() - start

        # print("save to file: " + sPfile)
        # os.makedirs(os.path.dirname(sPfile), exist_ok=True)
        # np.save(sPfile, sP)
        # np.save(sPWfile, sPW)
        # np.save(sNfile, sN)
        # np.save(sNWfile, sNW)
        # np.save(os.path.join(nds.dsDir, filePrefix + "-time.npy"), runTime)
    sP, sPW, sIP = selectTopkFast(nds.P, nds.PW, p_count, mdWeight=weightP * w)
    sN, sNW, sIN = selectTopkFast(nds.N, nds.NW, n_count, mdWeight=weightN * w)
    return ds.TrainSet(sP, sPW, sN, sNW, nds.dsDir)


def minDistInSet(idx, S, dist=distance.euclidean):
    """
    在S中找出离S[idx]最近的点

    Parameters
    ----------
    idx : TYPE
        DESCRIPTION.
    S : TYPE
        DESCRIPTION.
    dist : TYPE, optional
        DESCRIPTION. The default is distance.euclidean.

    Returns
    -------
    None.

    """
    minDis = math.inf
    p = S[idx]
    for i in range(len(S)):
        if i == idx:
            continue
        d = dist(p, S[i])
        if d < minDis:
            minDis = d
    return minDis


def totalProximitAndDiversity(P, dP, norm=True, dist=distance.euclidean):
    """
    计算TPD指标

    Parameters
    ----------
    P : TYPE
        P[i]：point i
    dP : TYPE
        dP[i]：negative of true distance to boundary for point i
    norm : TYPE, optional
        DESCRIPTION. The default is True.
    dist : TYPE, optional
        DESCRIPTION. The default is distance.euclidean.

    Returns
    -------
    totalP+totalMD：proximity + diversity的值
    totalP：proximity的值
    totalMD：diversity的值

    """

    totalP = 0
    totalMD = 0
    for i in range(len(P)):
        totalP += dP[i]
        totalMD += minDistInSet(i, P, dist=dist)

    if norm:
        return (totalP + totalMD) / len(P), totalP / len(P), totalMD / len(P)
    else:
        return (totalP + totalMD), totalP, totalMD


def datasetPD(P, dP, N, dN, norm=True, dist=distance.euclidean):
    TP, PP, MDP = totalProximitAndDiversity(P, dP, norm, dist)
    TN, PN, MDN = totalProximitAndDiversity(N, dN, norm, dist)

    return (TP + TN), (PP + PN), (MDP + MDN)


def computeMDWeightForProximity(ds, p):
    """
    1. 由于有些方法的proximity与diversity的量纲不同，不能直接相加。
    2. diversity的量纲是距离，在所有proximity方法中，只有out dist的量纲也是距离(可以与diversity直接相加)
    3. 除out dist外的其余proximity方法的量纲都不是距离，是介于0-1之间的数，因此需要对diversity做归一化，才能与proximity相加
        diversity归一化的方法是用diversity的值(距离)除以一个近似的平均距离(见util.py的randomAvgDist()方法)

    Parameters
    ----------
    ds : class
        数据集.
    p : function
        proximity方法.

    Returns
    -------
    1/ds.avgDistP：float, 对正类样本点diversity值进行归一化的权重
    1/ds.avgDistP：float, 对负类样本点diversity值进行归一化的权重
    """
    if p == proximity.computeClosenessToMarginByAvgOppKNNDist:
        return 1, 1
    else:
        if not hasattr(ds, "avgDistP"):
            ds.avgDistP = util.randomAvgDist(ds.P, max(100, int(len(ds.P) * 0.1)))

        if not hasattr(ds, "avgDistN"):
            ds.avgDistN = util.randomAvgDist(ds.N, max(100, int(len(ds.N) * 0.1)))

        return 1 / ds.avgDistP, 1 / ds.avgDistN


###############################################################################
# Unit Testing

import unittest


class TestSelectTopK(unittest.TestCase):
    def test_selectTopKFast01(self):
        P = np.array([3, 1, 5, 4, 7])
        PW = np.array([10, 8, 9, 7, 3])
        sP, sPW, sIP = selectTopkFast(P, PW, k=3, mdWeight=1)
        np.testing.assert_array_equal(sP, [3, 5, 1])
        np.testing.assert_array_equal(sPW, [10, 9, 8])
        np.testing.assert_array_equal(sIP, [0, 2, 1])

    def old_test_slowVSFast(self):  # Slow version has bug
        ts = ds.loadTwoCircle()
        pList = proximityNames.keys()

        r = 0.02
        p = list(pList)[0]
        pname = proximityNames[p]
        k = 10
        nds = addProximity(ts, k=k, clossnessCalc=p)
        sP, sPW, sIP = selectTopkFast(nds.P, nds.PW, int(r * len(nds.P)))
        sN, sNW, sIN = selectTopkFast(nds.N, nds.NW, int(r * len(nds.N)))

        sds = ds.TrainSet(sP, sPW, sN, sNW, nds.dsDir)

        sds.plot2D()
        plt.title("Fast: {0} k = {1} r = {2}".format(pname, k, r))
        circle1 = plt.Circle((0, 0), 0.75, color="r", fill=False, linewidth=2)
        plt.gcf().gca().add_artist(circle1)
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.show()

        nds = addProximity(ts, k=k, clossnessCalc=p)
        sP2, sPW2 = selectTopk(nds.P, nds.PW, int(r * len(nds.P)))
        sN2, sNW2 = selectTopk(nds.N, nds.NW, int(r * len(nds.N)))

        np.testing.assert_array_equal(sP, sP2)
        np.testing.assert_array_equal(sPW, sPW2)
        np.testing.assert_array_equal(sN, sN2)
        np.testing.assert_array_equal(sNW, sNW2)

    def oldtest_selectTopK02(self):
        P = np.array(
            [[0.20, -0.86], [-0.14, 0.41], [-0.36, 0.33], [-0.51, -0.13], [0.53, -0.08]]
        )
        N = np.array(
            [[-0.49, 0.37], [0.48, -0.93], [0.54, -0.28], [-0.63, -0.46], [0.31, 0.77]]
        )

        c10 = ds.TrainSet2Circle(dsDir="../test/two-circles-10", P=P, N=N, R=0.75)
        # PW = np.array([-0.1329496,  -0.31675642, -0.26163538, -0.22369211, -0.21399627])
        # NW = np.array([-0.13599674, -0.29656581, -0.14172375, -0.0300641,  -0.08006024])
        self.assertAlmostEqual(c10.PW[0], -0.1329496)
        self.assertAlmostEqual(c10.NW[-1], -0.080060239)

        def sortOppNeighborDesc(P, N, dist=distance.euclidean):
            PN = []
            for i in range(len(P)):
                d = np.zeros(len(N))
                I = np.zeros(len(N), dtype=int)
                for j in range(len(N)):
                    d[j] = dist(P[i], N[j])
                    I[j] = j
                print("i = ", i, " d: ", d)
                print("        idx: ", I)
                sI, sd = util.sortByKey(I, d, asc=True, returnKey=True)
                PN.append(list(zip(sI, sd)))
            print("neighbor (idx, d) asc: ", PN)
            return PN

        # PN = sortOppNeighborDesc(P, N)
        PN = [
            [
                (1.0, 0.2886173937932362),
                (2.0, 0.6723094525588644),
                (3.0, 0.9213576938409969),
                (0.0, 1.410319112825179),
                (4.0, 1.6337074401495513),
            ],
            [
                (0.0, 0.3522782990761707),
                (4.0, 0.5762811813689565),
                (2.0, 0.9687620966986683),
                (3.0, 0.9984988733093293),
                (1.0, 1.47648230602334),
            ],
            [
                (0.0, 0.13601470508735444),
                (4.0, 0.8015609770940698),
                (3.0, 0.8348652585896721),
                (2.0, 1.0872442227944925),
                (1.0, 1.5143315356948754),
            ],
            [
                (3.0, 0.35114099732158877),
                (0.0, 0.5003998401278722),
                (2.0, 1.0606601717798214),
                (4.0, 1.217538500417954),
                (1.0, 1.2728314892396402),
            ],
            [
                (2.0, 0.20024984394500786),
                (1.0, 0.8514693182963201),
                (4.0, 0.8780091115700337),
                (0.0, 1.114854250563723),
                (3.0, 1.2206555615733705),
            ],
        ]

        # NN = sortOppNeighborDesc(N, P)
        NN = [
            [
                (2.0, 0.13601470508735444),
                (1.0, 0.3522782990761707),
                (3.0, 0.5003998401278722),
                (4.0, 1.114854250563723),
                (0.0, 1.410319112825179),
            ],
            [
                (0.0, 0.2886173937932362),
                (4.0, 0.8514693182963201),
                (3.0, 1.2728314892396402),
                (1.0, 1.47648230602334),
                (2.0, 1.5143315356948754),
            ],
            [
                (4.0, 0.20024984394500786),
                (0.0, 0.6723094525588644),
                (1.0, 0.9687620966986683),
                (3.0, 1.0606601717798214),
                (2.0, 1.0872442227944925),
            ],
            [
                (3.0, 0.35114099732158877),
                (2.0, 0.8348652585896721),
                (0.0, 0.9213576938409969),
                (1.0, 0.9984988733093293),
                (4.0, 1.2206555615733705),
            ],
            [
                (1.0, 0.5762811813689565),
                (2.0, 0.8015609770940698),
                (4.0, 0.8780091115700337),
                (3.0, 1.217538500417954),
                (0.0, 1.6337074401495513),
            ],
        ]

        # outDistPW = []
        # for i in range(len(P)):
        #     outDistPW.append((-PN[i][0][1] - PN[i][1][1])/2)
        #     print("PW[{0}] = -({1} + {2})/2 = {3}".format(i,PN[i][0][1],PN[i][1][1],outDistPW[i]))
        # print("outDist.PW: ",outDistPW)
        outDistPW = [
            -0.4804634231760503,
            -0.4642797402225636,
            -0.46878784109071214,
            -0.4257704187247305,
            -0.525859581120664,
        ]

        # outDistNW = []
        # for i in range(len(N)):
        #     outDistNW.append((-NN[i][0][1] - NN[i][1][1])/2)
        #     print("NW[{0}] = -({1} + {2})/2 = {3}".format(i,NN[i][0][1],NN[i][1][1],outDistNW[i]))
        # print("outDist.NW: ",outDistNW)
        outDistNW = [
            -0.24414650208176258,
            -0.5700433560447782,
            -0.43627964825193616,
            -0.5930031279556305,
            -0.6889210792315131,
        ]

        nds = addProximity(
            c10, k=2, clossnessCalc=proximity.computeClosenessToMarginByAvgOppKNNDist
        )

        np.testing.assert_array_almost_equal(nds.PW, outDistPW)
        np.testing.assert_array_almost_equal(nds.NW, outDistNW)

        sP, sPW, sIP = selectTopkFast(nds.P, nds.PW, k=3, mdWeight=0)
        # print("selected P: ",sP)
        # print("selected PW: ",sPW)
        # print("selected idx: ",sIP)
        np.testing.assert_array_almost_equal(
            sP, [[-0.51, -0.13], [-0.14, 0.41], [-0.36, 0.33]]
        )
        np.testing.assert_array_almost_equal(
            sPW, [-0.4257704187247305, -0.4642797402225636, -0.46878784109071214]
        )
        np.testing.assert_array_almost_equal(sIP, [3, 1, 2])

        sN, sNW, sIN = selectTopkFast(nds.N, nds.NW, k=3, mdWeight=0)
        # print("selected P: ",sN)
        # print("selected PW: ",sNW)
        # print("selected idx: ",sIN)
        np.testing.assert_array_almost_equal(
            sN, [[-0.49, 0.37], [0.54, -0.28], [0.48, -0.93]]
        )
        np.testing.assert_array_almost_equal(
            sNW, [-0.24414650208176258, -0.43627964825193616, -0.5700433560447782]
        )
        np.testing.assert_array_almost_equal(sIN, [0, 2, 1])

    def test_selectTopK03(self):
        P = np.array(
            [[0.20, -0.86], [-0.14, 0.41], [-0.36, 0.33], [-0.51, -0.13], [0.53, -0.08]]
        )
        N = np.array(
            [[-0.49, 0.37], [0.48, -0.93], [0.54, -0.28], [-0.63, -0.46], [0.31, 0.77]]
        )

        c10 = ds.TrainSet2Circle(dsDir="../test/two-circles-10", P=P, N=N, R=0.75)
        # PW = np.array([-0.1329496,  -0.31675642, -0.26163538, -0.22369211, -0.21399627])
        # NW = np.array([-0.13599674, -0.29656581, -0.14172375, -0.0300641,  -0.08006024])
        self.assertAlmostEqual(c10.PW[0], -0.1329496)
        self.assertAlmostEqual(c10.NW[-1], -0.080060239)

        outDistPW = [
            -0.4804634231760503,
            -0.4642797402225636,
            -0.46878784109071214,
            -0.4257704187247305,
            -0.525859581120664,
        ]

        outDistNW = [
            -0.24414650208176258,
            -0.5700433560447782,
            -0.43627964825193616,
            -0.5930031279556305,
            -0.6889210792315131,
        ]

        nds = addProximity(
            c10, k=2, clossnessCalc=proximity.computeClosenessToMarginByAvgOppKNNDist
        )

        # d3 = []
        # for i in range(len(P)):
        #     d3.append(distance.euclidean(P[i],P[3]))
        # print("///////////////d3 ", d3)
        # print(nds.PW)
        # mdW = np.array(d3) + np.array(outDistPW)
        # print("mdW ",mdW)

        # d0 = []
        # for i in range(len(P)):
        #     d0.append(distance.euclidean(P[i],P[0]))
        # print("///////////////d0: ", d0)
        # md = np.minimum(d3,d0)
        # print("md...", md)
        # mdW = np.array(md) + np.array(outDistPW)
        # print("mdW Sec ",mdW)

        sP, sPW, sIP = selectTopkFast(nds.P, nds.PW, k=3, mdWeight=1)
        # print("selected P: ",sP)
        # print("selected PW: ",sPW)
        # print("selected idx: ",sIP)
        np.testing.assert_array_almost_equal(
            sP, [[-0.51, -0.13], [0.2, -0.86], [0.53, -0.08]]
        )
        np.testing.assert_array_almost_equal(
            sPW, [-0.4257704187247305, -0.4804634231760503, -0.525859581120664]
        )
        np.testing.assert_array_equal(sIP, [3, 0, 4])

        # print("我是分隔符================")
        # # 负样本
        # d0N = []
        # for i in range(len(N)):
        #     d0N.append(distance.euclidean(N[i],N[0]))
        # print("///////////////d0N ", d0N)
        # mdNW = np.array(d0N) + np.array(outDistNW)
        # print("mdNW ",mdNW)

        # d1N = []
        # for i in range(len(N)):
        #     d1N.append(distance.euclidean(N[i],N[1]))
        # print("///////////////d1N: ", d1N)
        # md = np.minimum(d0N,d1N)
        # print("md...", md)
        # mdW = np.array(md) + np.array(outDistNW)
        # print("mdW Sec ",mdW)

        sN, sNW, sIN = selectTopkFast(nds.N, nds.NW, k=3, mdWeight=1)
        # print("selected P: ",sN)
        # print("selected PW: ",sNW)
        # print("selected idx: ",sIN)
        np.testing.assert_array_almost_equal(
            sN, [[-0.49, 0.37], [0.48, -0.93], [-0.63, -0.46]]
        )
        np.testing.assert_array_almost_equal(
            sNW, [-0.24414650208176258, -0.5700433560447782, -0.5930031279556305]
        )
        np.testing.assert_array_almost_equal(sIN, [0, 1, 3])


# =============================================================================
# 实验
# =============================================================================
def plotTrueProximityDeversity(ts, rList, pkList, wList=[1], dist=distance.euclidean):
    PD = np.zeros((len(rList), len(pkList), len(wList), 3))

    import time

    startTime = time.time()

    for b, (p, k) in enumerate(pkList):
        nds = addProximity(ts, k=k, clossnessCalc=p)
        weightP, weightN = computeMDWeightForProximity(ts, p)
        print("weightP: {0}; weightN: {1}".format(weightP, weightN))

        for a, r in enumerate(rList):
            for c, w in enumerate(wList):
                sP, sPW, sIP = selectTopkFast(
                    nds.P, nds.PW, int(r * len(nds.P)), mdWeight=weightP * w
                )
                sN, sNW, sIN = selectTopkFast(
                    nds.N, nds.NW, int(r * len(nds.N)), mdWeight=weightN * w
                )

                # sds = ds.TrainSet(sP, sPW, sN, sNW, nds.dsDir)
                dP = ts.PW[sIP]
                dN = ts.NW[sIP]

                PD[a, b, c] = datasetPD(sP, dP, sN, dN, norm=True, dist=dist)
                print(
                    "p: {0}/{1}; r:{2}/{3}; w:{4}/{5} done, time: {6} s".format(
                        b,
                        len(pkList),
                        a,
                        len(rList),
                        c,
                        len(wList),
                        time.time() - startTime,
                    )
                )

    for a, r in enumerate(rList):  # 每个r一幅图
        for b, (p, k) in enumerate(pkList):  # 每个方法一个条线
            pname = proximityNames[p] + "(k=" + str(k) + ")"
            plt.plot(wList, PD[a, b, :, 0], label=pname)
            plt.plot(
                wList, PD[a, b, :, 1], linestyle="dashed", label=pname + "(_-d[i])"
            )
        plt.legend()
        plt.title("True Proximity Diversity, r = " + str(r))
        plt.ylabel("TPD")
        plt.xlabel("w")
        plt.show()


# =============================================================================
# 对多组数据(生成数据的随机种子不同)得到的PD求平均和标准方差，并且进行绘图
# =============================================================================
def TPDByWOverdsLists(dsList, rList, pkList, wList, dist=distance.euclidean):
    PD = np.zeros((len(rList), len(pkList), len(wList), 3, len(dsList)))  # TPD

    for dsIdx, dset in enumerate(dsList):
        for b, (p, k) in enumerate(pkList):
            nds = addProximity(dset, k=k, clossnessCalc=p)

            weightP, weightN = computeMDWeightForProximity(dset, p)
            print("weightP: {0}; weightN: {1}".format(weightP, weightN))

            for a, r in enumerate(rList):
                for c, w in enumerate(wList):
                    sP, sPW, sIP = selectTopkFast(
                        nds.P, nds.PW, int(r * len(nds.P)), mdWeight=weightP * w
                    )
                    sN, sNW, sIN = selectTopkFast(
                        nds.N, nds.NW, int(r * len(nds.N)), mdWeight=weightN * w
                    )

                    # sds = ds.TrainSet(sP, sPW, sN, sNW, nds.dsDir)
                    dP = dset.PW[sIP]
                    dN = dset.NW[sIP]

                    PD[a, b, c, 0, dsIdx] = datasetPD(
                        sP, dP, sN, dN, norm=True, dist=dist
                    )[
                        0
                    ]  # 记录proximity + diversity的信息
                    PD[a, b, c, 1, dsIdx] = datasetPD(
                        sP, dP, sN, dN, norm=True, dist=dist
                    )[
                        1
                    ]  # 只记录proximity的信息
                    PD[a, b, c, 2, dsIdx] = datasetPD(
                        sP, dP, sN, dN, norm=True, dist=dist
                    )[
                        2
                    ]  # 只记录diversity的信息
                    print(
                        "dset:{0}/{1}, pk:{2}/{3}, r:{4}/{5}, w:{6}/{7} done".format(
                            dsIdx,
                            len(dsList),
                            b,
                            len(pkList),
                            a,
                            len(rList),
                            c,
                            len(wList),
                        )
                    )
    print("##############", PD.shape)
    return (len(dsList), rList, pkList, wList, PD)


def plotAvg_CTDByW(dsCount, rList, pkList, wList, PD, ylim=None):
    """
    对多组数据(生成数据的随机种子不同)得到PD求平均和标准方差，并且进行绘图
    """
    print("shape of PD", PD.shape)
    for a, r in enumerate(rList):  # 每个r一幅图
        for b, (p, k) in enumerate(pkList):  # 每个方法一个条线
            pname = proximityNames[p] + "(k=" + str(k) + ")"
            avg = PD.mean(axis=4, keepdims=True)  # 沿着dsList这个维度求PD的平均，得到的avg仍然是五维
            avg = np.array(avg)
            print("**********shape of avg", avg.shape)
            var = PD.var(axis=4, keepdims=True)
            std = np.sqrt(var)

            # 画 proximity + diversity
            plt.plot(wList, avg[a, b, :, 0, :], label=pname)
            plt.plot(
                wList,
                avg[a, b, :, 0, :] + std[a, b, :, 0, :],
                label=pname + "+std",
                linestyle="dashed",
            )
            plt.plot(
                wList,
                avg[a, b, :, 0, :] - std[a, b, :, 0, :],
                label=pname + "-std",
                linestyle="dashed",
            )

            # 画 diversity
            plt.plot(wList, avg[a, b, :, 1, :], label=pname, linestyle="dotted")
            plt.plot(
                wList,
                avg[a, b, :, 1, :] + std[a, b, :, 1, :],
                label=pname + "+std",
                linestyle="dashdot",
            )
            plt.plot(
                wList,
                avg[a, b, :, 1, :] - std[a, b, :, 1, :],
                label=pname + "-std",
                linestyle="dashdot",
            )

        plt.legend()
        plt.title(
            "True Proximity Diversity, r ={0} over {1} datasets".format(r, dsCount)
        )
        plt.ylabel("TPD")
        plt.xlabel("w")
        plt.show()


def avg_TPDByW(
    dsList,
    resultFile,
    dsCount=5,
    rList=[0.2],
    pkList=[
        (proximity.computeClosenessToMarginByAlternatingChain, 10),
        (proximity.computeClosenessToMarginByAvgOppKNNDist, 20),
    ],
    ylim=(0, 2),
    wList=[i * 0.5 for i in range(11)],
):
    print("&&&&&&&& rList Length", len(rList))
    result = util.computeOrLoad(
        resultFile, lambda: TPDByWOverdsLists(dsList, rList, pkList, wList)
    )

    plotAvg_CTDByW(*result)


# 2C0.2
def exp_2020Jul10_searchForW_2C02(
    dirName="../test/selectTopK-2020-07-10-2C0.2",
    dsCount=20,
    rList=[0.05, 0.1, 0.2, 0.3],
):
    dsList = [ds.loadTwoCircleNoise(seed=s) for s in range(dsCount)]
    avg_TPDByW(
        dsList,
        os.path.join(dirName, "result10-100-ds20.npy"),
        dsCount=dsCount,
        rList=rList,
    )


# 2C0.2_nr
def exp_2020Jul11_searchForW_2C02_nr02(
    dirName="../test/selectTopK-2020-07-11-2C0.2nr02",
    dsCount=20,
    rList=[0.05, 0.1, 0.2, 0.3],
):
    dsList = [ds.loadTwoCircleNoise(seed=s) for s in range(dsCount)]
    dsNrList = []
    for i, ts in enumerate(dsList):
        dsNrList.append(removeNoiseForCircle(ts, 0.2, 0.2))
        print("noise removal for i: {0}/{1} done. {2}".format(i, dsCount, ts.dsDir))

    avg_TPDByW(
        dsNrList,
        os.path.join(dirName, "result10-100-ds20.npy"),
        dsCount=dsCount,
        rList=rList,
    )


# 2C0.1
def exp_2020Jul11_searchForW_2C01(
    dirName="../test/selectTopK-2020-07-11-2C0.1",
    dsCount=20,
    rList=[0.05, 0.1, 0.2, 0.3],
):
    dsList = [ds.loadTwoCircleNoise(noise=0.1, seed=s) for s in range(dsCount)]
    avg_TPDByW(
        dsList,
        os.path.join(dirName, "result10-100-ds20.npy"),
        dsCount=dsCount,
        rList=rList,
    )


# 2G0.7
def exp_2020Jul11_searchForW_2G07(
    dirName="../test/selectTopK-2020-07-11-2G0.7",
    dsCount=20,
    rList=[0.05, 0.1, 0.2, 0.3],
):
    dsList = [ds.loadTwoGuasianNoise(seed=s) for s in range(dsCount)]
    avg_TPDByW(
        dsList,
        os.path.join(dirName, "result10-100-ds20.npy"),
        dsCount=dsCount,
        rList=rList,
    )


# 2G0.7nr01
def exp_2020Jul11_searchForW_2G07nr01(
    dirName="../test/selectTopK-2020-07-11-2G0.7nr01",
    dsCount=20,
    rList=[0.05, 0.1, 0.2, 0.3],
):
    dsList = [ds.loadTwoGuasianNoise(seed=s) for s in range(dsCount)]
    dsNrList = []
    for i, ts in enumerate(dsList):
        dsNrList.append(removeNoiseFor2Guasian(ts, 0.2, 0.2))
        print("noise removal for i: {0}/{1} done. {2}".format(i, dsCount, ts.dsDir))
    avg_TPDByW(
        dsNrList,
        os.path.join(dirName, "result10-100-ds20.npy"),
        dsCount=dsCount,
        rList=rList,
    )


def plotTopKFor2Circle(ts, rList, pkList, wList):
    if not isinstance(ts, ds.TrainSet2Circle):
        raise RuntimeError("ts must be TrainSet2Circle")

    for r in rList:
        for p, k in pkList:
            weightP, weightN = computeMDWeightForProximity(ts, p)
            print("weightP: {0}; weightN: {1}".format(weightP, weightN))

            pname = proximityNames[p]

            for w in wList:
                nds = addProximity(ts, k=k, clossnessCalc=p)
                sP, sPW, sIP = selectTopkFast(
                    nds.P, nds.PW, int(r * len(nds.P)), mdWeight=weightP * w
                )
                sN, sNW, sIN = selectTopkFast(
                    nds.N, nds.NW, int(r * len(nds.N)), mdWeight=weightN * w
                )

                sds = ds.TrainSet(sP, sPW, sN, sNW, nds.dsDir)

                sds.plot2D()

                plt.title(
                    "{0} k = {1} r = {2} w={3} data set: {4}".format(
                        pname, k, r, w, ts.dsName
                    )
                )
                circle1 = plt.Circle((0, 0), 0.75, color="r", fill=False, linewidth=2)
                plt.gcf().gca().add_artist(circle1)
                plt.xlim(-1.2, 1.2)
                plt.ylim(-1.2, 1.2)
                plt.show()


def expSearchForW():
    ts = ds.loadTwoCircleNoise()
    ts.dsName = "2C0.2"
    ac = proximity.computeClosenessToMarginByAlternatingChain
    outDist = proximity.computeClosenessToMarginByAvgOppKNNDist
    rList = [0.2]

    pkList = [(ac, 10), (outDist, 10)]
    wList = [i * 0.5 for i in range(0, 11, 1)]
    plotTrueProximityDeversity(ts, rList=rList, pkList=pkList, wList=wList)
    #    plotTopKFor2Circle(ts, rList=rList, pkList = [(outDist,10)], wList=wList)
    #    plotTopKFor2Circle(ts, rList=rList, pkList = [(ac,10)], wList=wList)

    ts = removeNoiseForCircle(ts, 0.2, 0.2)
    ts.dsName = "2C0.2-nr"
    pkList = [(ac, 4), (outDist, 10)]
    plotTrueProximityDeversity(ts, rList=rList, pkList=pkList, wList=wList)


#    plotTopKFor2Circle(ts, rList=rList, pkList = [(outDist,10)], wList=wList)
#    plotTopKFor2Circle(ts, rList=rList, pkList = [(ac,4)], wList=wList)


def acW2vs4_2Cnr():
    ts = ds.loadTwoCircleNoise()
    ts.dsName = "2C0.2"
    ac = proximity.computeClosenessToMarginByAlternatingChain
    rList = [0.2]

    ts = removeNoiseForCircle(ts, 0.2, 0.2)
    ts.dsName = "2C0.2-nr"
    plotTopKFor2Circle(ts, rList=rList, pkList=[(ac, 4)], wList=[2, 4])


def showImpactOfW_2Cnr():
    ts = ds.loadTwoCircleNoise()
    ts.dsName = "2C0.2"
    ac = proximity.computeClosenessToMarginByAlternatingChain
    outDist = proximity.computeClosenessToMarginByAvgOppKNNDist
    rList = [0.2]

    ts = removeNoiseForCircle(ts, 0.1, 0.1)
    ts.dsName = "2C0.1-nr"
    plotTopKFor2Circle(ts, rList=rList, pkList=[(ac, 4)], wList=[0, 1, 2])
    plotTopKFor2Circle(ts, rList=rList, pkList=[(outDist, 10)], wList=[0, 1, 1.5])


def outdistImpactOfDeveristy():
    ts = ds.loadTwoCircleNoise()
    ts.dsName = "2C0.2"
    outDist = proximity.computeClosenessToMarginByAvgOppKNNDist
    rList = [0.2]

    plotTopKFor2Circle(ts, rList=rList, pkList=[(outDist, 10)], wList=[0, 2])

    ts = removeNoiseForCircle(ts, 0.2, 0.2)
    ts.dsName = "2C0.2-nr"
    plotTopKFor2Circle(ts, rList=rList, pkList=[(outDist, 10)], wList=[0, 1.5])


def plotCircle():
    ts = ds.loadTwoCircleNoise()
    ts.dsName = "2C0.2"
    ac = proximity.computeClosenessToMarginByAlternatingChain
    outDist = proximity.computeClosenessToMarginByAvgOppKNNDist
    rList = [0.2]

    pkList = [(ac, 10), (outDist, 10)]
    wList = [i * 0.5 for i in range(0, 11, 1)]
    plotTopKFor2Circle(ts, rList=rList, pkList=[(outDist, 10)], wList=[0, 1, 2])
    plotTopKFor2Circle(ts, rList=rList, pkList=[(ac, 10)], wList=[0, 2, 4])

    ts = removeNoiseForCircle(ts, 0.2, 0.2)
    ts.dsName = "2C0.2-nr"
    pkList = [(ac, 4), (outDist, 10)]
    plotTopKFor2Circle(ts, rList=rList, pkList=[(outDist, 10)], wList=[0, 1, 1.5])
    plotTopKFor2Circle(ts, rList=rList, pkList=[(ac, 4)], wList=[0, 1, 2])


if __name__ == "__main__":
    # unittest.main()

    # expSearchForW()
    # acW2vs4_2Cnr()
    showImpactOfW_2Cnr()
    # exp_2020Jul10_searchForW_2C02()
    # exp_2020Jul11_searchForW_2C02_nr02
    # exp_2020Jul11_searchForW_2C01()
    # exp_2020Jul11_searchForW_2C02_nr02()
    # exp_2020Jul11_searchForW_2G07()
    # exp_2020Jul11_searchForW_2G07nr01()
    # from playsound import playsound
    # playsound('../Canon.mp3')
