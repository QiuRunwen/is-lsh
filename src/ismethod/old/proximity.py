# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 09:10:11 2020

@author: admin
定义proximity的四种方法：
1. AlternativeChain
2. FreqeucyInOppKNN
3. 程序出错：ReversedAlternativeChain
4. 排序分数很容易一致，导致区分度不高，改为邻居的平均距离 ：AvgOppKNNDist
"""

import operator
import os
import warnings

import numpy as np
from scipy.spatial import distance

from . import dataset as ds
from .util import computeOrLoad, run_time, sortByKey


def KNN_allow_self(q, P, k, dist=distance.euclidean):
    """
    给定一个点 q, 在 P 中找出 k 个离 q 点最近的点
    dist: 计算两个点之间的距离的函数

    Parameters
    ----------
    q : array like
        reference point
    P : P[i] is array like
        an array/list of points
    k : int
    dist : distance metric, optional
        d(p,q) give a positive float indicate the distance between two point p and q. The default is distance.euclidean.
    Returns
    -------
        r = [(i_0,d_0),(i_1,d_1),(i_2,d_2),...]
        indicating P[i_j] is j-th neariest neighor of q, with distance d_j, j=0,1,...,k-1
    """
    if k > len(P):
        warnings.warn("Warning: 邻居个数k: %s 超过数据集大小: %s" % (k, len(P)))
        k = len(P)

    d = np.zeros(len(P)) * np.nan

    for i in range(len(P)):
        d[i] = dist(P[i], q)

    idx = np.argpartition(d, k)[:k]  # 用argpartition找出最小的k个distance的下标
    dist = d[idx]

    sIdx, sDist = sortByKey(idx, dist, asc=True, returnKey=True)

    #    r = list(zip(idx,dist))
    #    r = sorted(r, key=operator.itemgetter(1),reverse = False)   # 对邻居按距离升序排列，在r中越靠前的，表示距离越近
    return list(zip(sIdx, sDist))


def KNN(q, P, k, dist=distance.euclidean, q_idx=None):
    """
    给定一个点 q, 在 P 中找出 k 个离 q 点最近的点
    dist: 计算两个点之间的距离的函数

    Parameters
    ----------
    q : array like
        reference point
    P : P[i] is array like
        an array/list of points
    k : int
    dist : distance metric, optional
        d(p,q) give a positive float indicate the distance between two point p and q. The default is distance.euclidean.
    q_idx : int
        index of q in P. If not None, when computing kNN will ignor q itself.
    Returns
    -------
        r = [(i_0,d_0),(i_1,d_1),(i_2,d_2),...]
        indicating P[i_j] is j-th neariest neighor of q, with distance d_j, j=0,1,...,k-1
    """

    if q_idx is None:
        return KNN_allow_self(q, P, k, dist)

    r = []
    for idx, dist in KNN_allow_self(q, P, k + 1, dist):
        if idx != q_idx:
            r.append((idx, dist))
    if len(r) > k:
        return r[:k]
    else:
        return r


def computeOppositeNeighbors(P, N, k, dist=distance.euclidean):
    """
    为 P 中的每个点找到 N 中的最近 k 个邻居
    P: P[i] 表示第i个正样本
    N: N[i] 表示第i个负样本
    为每个正样本找到 k 个最近的负样本
    返回 PN:
       PN[l] = [(i_0,d_0),(i_1,d_1),...]
       N[i_j] 是离 P[l] 第 j 近的邻居，距离为 d_j, j = 0, 1, ..., k-1
    """
    PN = []
    import time

    startTime = time.process_time()
    nextPrintTime = 0
    for l in range(len(P)):
        i = KNN(P[l], N, k, dist)
        PN.append(i)

        elapsedTime = time.process_time() - startTime
        if elapsedTime >= nextPrintTime:
            # print(
            #     "find KNN for {0}/{1}, elapsed time: {2} s".format(
            #         l, len(P), elapsedTime
            #     )
            # )
            nextPrintTime += 30

    return PN


def computeAllNeighbors(P, N, k, dist=distance.euclidean):
    """
    为 A = P + N 中的每个点找到 A 中(不包括自己)的最近 k 个邻居
    正样本用本身的索引，负样本 j 的统一索引为 j + len(P)

    AN[l] = [(i_0,d_0),(i_1,d_1),...]
        A[i_j] 为 A[l] 第 j 近的邻居，距离为 d_j
    """
    A = np.vstack((P, N))
    AN = []
    import time

    startTime = time.process_time()
    nextPrintTime = 0
    for l in range(len(A)):
        i = KNN(q=A[l], P=A, k=k, dist=dist, q_idx=l)
        AN.append(i)

        elapsedTime = time.process_time() - startTime
        if elapsedTime >= nextPrintTime:
            # print(
            #     "find KNN for {0}/{1}, elapsed time: {2} s".format(
            #         l, len(A), elapsedTime
            #     )
            # )
            nextPrintTime += 30

    return AN


# old name: createOrLoadOppKNN
def createOrLoadKNN(ds, k, dist=distance.euclidean):
    """
    按照指定距离公式 dist 计算每个样本
    给定 X,y,k，根据不同距离公式计算正负样本点的k个最近邻居并保存到文件

    Parameters
    ----------
    X : 二维或以上数值
        x[i] 表示第i个数据点
    y : 一维向量
        y[i] 表示第i个数据点的标签，0或1
    k : 正整数
        最近的k个异类邻居
    dsDir : 字符串
        保存数据的目录
    dist : 距离函数
        The default is distance.euclidean.

    Returns
    -------
        P,PN,N,NN

    """

    P = ds.P
    N = ds.N

    if k > len(P):
        raise RuntimeError("正样本个数：{0} 小于k：{1}".format(len(P), k))
    if k > len(N):
        raise RuntimeError("负样本个数：{0} 小于k：{1}".format(len(N), k))

    prefix = dist.__name__ + "-"
    maxK = searchLargestK(ds.dsDir, prefix)

    if maxK < k:
        maxK = k
    
    neighborDir = os.path.join(ds.dsDir, prefix + str(maxK))

    AN = computeOrLoad(
        os.path.join(neighborDir, "AN.npy"), lambda: computeAllNeighbors(P, N, k, dist)
    )
    PN = computeOrLoad(
        os.path.join(neighborDir, "PN.npy"),
        lambda: computeOppositeNeighbors(P, N, k, dist),
    )
    NN = computeOrLoad(
        os.path.join(neighborDir, "NN.npy"),
        lambda: computeOppositeNeighbors(N, P, k, dist),
    )

    if len(P) + len(N) != len(AN):
        raise RuntimeError(
            f"{len(P) + len(N)=} != {len(AN)}, number of positive samples changed since last save. Check if train data is changed. Possibly by different random seed during generation or sampling"
        )
    if len(P) != len(PN):
        raise RuntimeError(
            f"{len(P)=} != {len(PN)}, number of positive samples changed since last save. Check if train data is changed. Possibly by different random seed during generation or sampling"
        )
    if len(N) != len(NN):
        raise RuntimeError(
            f"{len(N)=} != {len(NN)}, number of negative samples changed since last save. Check if train data is changed. Possibly by different random seed during generation or sampling"
        )

    return PN, NN, AN


def findFirstNewNeighbor(neighbors, I, idx):
    """
    从 neighbors 中找到第一个不在 I[0:idx] 中的邻居
    Return:
        (j,dj)
    """

    for j, dj in neighbors:
        j = int(j)
        if j not in I[0:idx]:
            return (j, dj)


def findLastNewNeighbor(neighbors, I, idx):
    """
    从 neighbors 中找到第一个不在 I[0:idx] 中的邻居
    Return:
        (j,dj)
    """

    for j, dj in reversed(neighbors):
        j = int(j)
        if j not in I[0:idx]:
            return (j, dj)


def findAlternatingChain(NS, NO, i, k, nextN=findFirstNewNeighbor):
    """
    根据N,P,NN,PN计算每个点的 chain ,为第 i 个点找一个长度为 2k+1 的 chain
    NS:与 i 同类的点，每个点记录 k+1 或以上个邻居
    NO:与 i 异类的点，每个点记录 k+1 或以上个邻居
    Return:
        [i_1,i_2,...i_(2k+1)]
        [d_1,d_2,...d_(2k)]
    """
    E = np.zeros(k + 1)  # 存所有与i是同类的点

    E[0] = i
    elen = 1

    Odd = np.zeros(k)  # 存所有与i是异类的点
    olen = 0

    d = np.zeros(2 * k)  # *np.nan    # 2k+1个点之间的两两距离

    i = int(i)
    neighbors = NS[i]  # 当前点的最近k个异类点的邻居

    if len(neighbors) <= k:
        raise RuntimeError("NS[{0}]邻居数：{1} 没有达到k:{2}+1".format(i, len(neighbors), k))

    for idx in range(1, 2 * k + 1):
        # 在neighbor中找到第一个不在I[0:idx]中的点
        if idx % 2 == 0:
            # print("idx: "+str(idx))
            # print("neighbors: ")
            # print(neighbors)
            # print("E:")
            # print(E[0:elen])

            (j, dj) = nextN(neighbors, E, elen)  # bug: 之前用 idx， 会在数组尾部填很多 0
            #            print("found: {0}".format((j,dj)))
            E[elen] = j
            elen += 1
        else:
            # print("idx: "+str(idx))
            # print("neighbors: ")
            # print(neighbors)
            # print("O:")
            # print(Odd[0:olen])

            (j, dj) = nextN(neighbors, Odd, olen)  # bug: 之前用 idx， 会在数组尾部填很多 0
            #            print("found: {0}".format((j,dj)))
            Odd[olen] = j
            olen += 1

        d[idx - 1] = dj
        neighbors = NO[j]
        (NS, NO) = (NO, NS)

    return E, Odd, d


# @article{zhu2016extended,
#   title={Extended nearest neighbor chain induced instance-weights for SVMs},
#   author={Zhu, Fa and Yang, Jian and Gao, Junbin and Xu, Chunyan},
#   journal={Pattern Recognition},
#   volume={60},
#   pages={863--874},
#   year={2016},
#   publisher={Elsevier}
# }
def computeClosenessToMarginByAlternatingChain(
    C, SN, ON, k, nextN=findFirstNewNeighbor, *other_lst, **other_dict
):
    """
    C: 同类样本
    SN: 与 C 同类的，每个样本的 k+1 个或以上最近异类邻居
    ON: 与 C 异类的，每个样本的 k+1 个或以上最近异类邻居

    按照点到 Margin 从近到远对样本 C 排序
    评估一个点到 Margin 的“近”用 PR 论文中的 Alternating Chain 的方法
    返回 [(i1,w1),(i2,w2),...]
        其中(ij,wj) 表示 C[ij] 是距离 Margin 第 j 近的点
    """
    Pweight = []
    for i in range(len(C)):
        E, Odd, d = findAlternatingChain(SN, ON, i, k, nextN=nextN)

        # 给定一个Chain中的(I,d),计算出每个点的权重
        margin = (sum(d) - max(d)) / (len(d) - 1)
        weight = margin / max(d)
        Pweight.append(weight)

    return np.array(Pweight)


def computeClosenessToMarginByAlternatingChainReversed(
    C, SN, ON, k, *other_lst, **other_dict
):
    return computeClosenessToMarginByAlternatingChain(
        C, SN, ON, k, nextN=findLastNewNeighbor
    )


# @inproceedings{shin2002pattern,
#   title={Pattern selection for support vector classifiers},
#   author={Shin, Hyunjung and Cho, Sungzoon},
#   booktitle={International Conference on Intelligent Data Engineering and Automated Learning},
#   pages={469--474},
#   year={2002},
#   organization={Springer}
# }
def computeClosenessToMarginByEntropy(
    C, SN, ON, k, AN, offset, *other_lst, **other_dict
):
    """
    C: 一个类的所有样本
    SN: 与 C 同类的，每个样本的 k 个或以上最近异类邻居
    ON: 与 C 异类的，每个样本的 k 个或以上最近异类邻居
    AN: 每个样本在所有样本中的 k 个最近邻居（不包括自己）, AN[l+offset] is kNN for C[l]
    k：
    """
    if offset == 0:  # C is positive class
        len_P = len(C)
    else:  # C is negative class
        len_P = offset

    neg_entropy = np.zeros(len(C))
    for i in range(len(C)):
        neighbor = AN[i + offset][:k]
        p_count = 0  # neighbors from positive class
        n_count = 0  # neighbors from negative class
        for idx, d in neighbor:
            if idx < len_P:
                p_count += 1
            else:
                n_count += 1

        entropy = 0
        if p_count != 0:
            p = p_count / k
            entropy += p * np.log2(1 / p)
        if n_count != 0:
            p = n_count / k
            entropy += p * np.log(1 / p)

        if offset == 0:
            same_class_count = p_count
        else:
            same_class_count = n_count

        if same_class_count >= 0.5 * k:
            neg_entropy[
                i
            ] = -entropy  # samples with larger entropy are closer to decision boundary
        else:  # samples with same_class_count/k < 0.5 are skipped by setting its closenes to infity
            neg_entropy[i] = np.infty

    return neg_entropy


# @article{zhu2020nearcount,
#   title={NearCount: Selecting critical instances based on the cited counts of nearest neighbors},
#   author={Zhu, Zonghai and Wang, Zhe and Li, Dongdong and Du, Wenli},
#   journal={Knowledge-Based Systems},
#   volume={190},
#   pages={105196},
#   year={2020},
#   publisher={Elsevier}
# }
#
# For a sample x in class C, its citation count is the number of samples x'
# outside class C whose kNN in class C contains x.
# Compute citation count normalized by maximum citation count.
def computeClosenessToMarginByFreqeucyInOppKNN(
    C, SN, ON, k, chainReverse=False, *other_lst, **other_dict
):
    """
    C: 一个类的样本
    SN: 与 C 同类的，每个样本的 k 个或以上最近异类邻居
    ON: 与 C 异类的，每个样本的 k 个或以上最近异类邻居
    k：

    一个 C 类样本 x 的引用者是 C 类以外的样本 x'， x'在 C 类中的最近 k 个邻居包含 x
    Return：
        计算每个样本的引用者数量，然后除以引用者数量的最大值变成 0-1 之间的数
    """

    count = np.zeros(len(C))
    for neighbors in ON:
        for idx, d in neighbors[:k]:
            idx = int(idx)
            count[idx] = count[idx] + 1

    return count / max(count)


def computeClosenessToMarginByAvgOppKNNDist(C, SN, ON, k, *other_lst, **other_dict):
    """
    C: 一个类型的样本
    SN: 与 C 同类的，每个样本的 k 个或以上最近异类邻居
    ON: 与 C 异类的，每个样本的 k 个或以上最近异类邻居
    k：

    计算每个点的k个异类邻居的平均距离，平均距离越大，离边界越远
    Argument:
        ON: (NN/PN)每个点传进去的异类邻居
    Return：
        第i个点被异类点算进邻居的次数，用最大值进行归一化
    """

    oppKnnDist = np.zeros(len(C))

    for i in range(len(SN)):
        neighbors = SN[i][:k]
        oppKnnDist[i] = sum(np.array(neighbors)[:, 1])
    return -oppKnnDist / k


@run_time
def searchLargestK(dirName, prefix):
    """
    在dirName目录下找所有prefix开头的文件或目录名: prefix-xxx，找到最大的整数xxx
    Parameters
    ----------
    dirName : 字符串
        目录名字
    prefix : 字符串
        要找的文件或目录的前缀

    Returns
    -------
    maxK : int
        文件名或目录名字中最大的后缀整数

    """
    maxK = -1
    if dirName is not None:
        if not os.path.exists(dirName):
            return maxK

    prefixLen = len(prefix)
    for file in os.listdir(dirName):
        if file.startswith(prefix):
            try:
                k = int(file[prefixLen:])
                if k > maxK:
                    maxK = k
            except ValueError:
                pass
    return maxK


def getNeighborCount(k, clossnessCalc=computeClosenessToMarginByFreqeucyInOppKNN):
    if clossnessCalc == computeClosenessToMarginByAlternatingChain:
        return k + 1
    return k


@run_time
def createOrLoadCloseness(
    ds,
    k,
    clossnessCalc=computeClosenessToMarginByFreqeucyInOppKNN,
    dist=distance.euclidean,
):
    """
    给定X,y,k，根据不同的计算方法以及距离公式计算正负样本点到margin的距离
    """

    PN, NN, AN = createOrLoadKNN(
        ds, getNeighborCount(k, clossnessCalc=clossnessCalc), dist=dist
    )

    neighborDir = os.path.join(ds.dsDir, dist.__name__ + "-" + str(k))
    closeDir = os.path.join(neighborDir, clossnessCalc.__name__)

    PW = computeOrLoad(
        os.path.join(closeDir, "PW.npy"),
        lambda: clossnessCalc(ds.P, PN, NN, k, AN=AN, offset=0),
    )
    NW = computeOrLoad(
        os.path.join(closeDir, "NW.npy"),
        lambda: clossnessCalc(ds.N, NN, PN, k, AN=AN, offset=len(ds.P)),
    )

    return PW, NW


def addProximity(
    ts,
    k,
    clossnessCalc=computeClosenessToMarginByFreqeucyInOppKNN,
    dist=distance.euclidean,
):
    PW, NW = createOrLoadCloseness(ts, k=k, clossnessCalc=clossnessCalc, dist=dist)
    if isinstance(ts, ds.TrainSetWithGT):
        nts = None
        if isinstance(ts, ds.TrainSet2Guasian):
            nts = ds.TrainSet2Guasian(ts.dsDir, ts.P, ts.N)
        if isinstance(ts, ds.TrainSet2Circle):
            nts = ds.TrainSet2Circle(ts.dsDir, ts.P, ts.N, ts.R)
        if isinstance(ts, ds.TrainSetXOR):
            nts = ds.TrainSetXOR(ts.dsDir, ts.P, ts.N)

        if nts != None:
            nts.PW = PW
            nts.NW = NW
            return nts
        else:
            raise RuntimeError(
                "Please implement for data set type: {0}".format(type(ts))
            )

    return ds.TrainSet(ts.P, PW, ts.N, NW, ts.dsDir)


proximityNames = {
    computeClosenessToMarginByAlternatingChain: "AC",
    computeClosenessToMarginByAlternatingChainReversed: "ACR",
    computeClosenessToMarginByFreqeucyInOppKNN: "Freq",
    computeClosenessToMarginByAvgOppKNNDist: "out dist",
    computeClosenessToMarginByEntropy: "entropy",
}


###############################################################################
# Unit Testing

import unittest


class TestFindMargin(unittest.TestCase):
    def test_searchLargestK(self):
        dsDir = "D:/Study/icu-ai.git/fuying/test/"
        prefix = "euclidian-"

        dir1 = os.path.join(dsDir, prefix + "10")
        if not os.path.exists(dir1):
            os.makedirs(dir1)
        dir2 = os.path.join(dsDir, prefix + "20")
        if not os.path.exists(dir2):
            os.makedirs(dir2)

        maxK = searchLargestK(dsDir, prefix)
        self.assertEqual(maxK, 20)

        os.rmdir(dir2)
        os.rmdir(dir1)

    def assert_indices_set_equal(idx_dist_pairs, idx_lst):
        idx_set = set()
        for idx, dist in idx_dist_pairs:
            idx_set.add(idx)
        assert len(idx_dist_pairs) == len(idx_set)
        exp_set = set(idx_lst)
        assert idx_set == exp_set

    def test_KNN(self):
        P = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
        a = KNN([0, 0], P, 3)
        TestFindMargin.assert_indices_set_equal(a, [0, 3, 1])
        b = KNN(P[0], P, 3, q_idx=0)
        TestFindMargin.assert_indices_set_equal(b, [4, 3, 1])


if __name__ == "__main__":
    unittest.main()

    # motiving example
    import dataset as ds
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    c = ds.loadTwoCircleNoise(noise=0.15, seed=90, n_samples=500)
    plt.style.use("ggplot")
    plt.rc("font", family="Times New Roman")

    plt.scatter(c.P[:, 0], c.P[:, 1], label="class 1")
    plt.scatter(c.N[:, 0], c.N[:, 1], label="class 0")

    from sklearn.svm import SVC

    X, y = c.toXY()
    clf = SVC(C=1, gamma=1)
    clf.fit(X, y)
    clf.fit(X, y)
    minX = -1
    maxX = 1
    minY = -1
    maxY = 1
    plt.xlim((minX - 0.2, maxX + 0.2))
    plt.ylim((minY - 0.2, maxY + 0.2))
    xx = np.linspace(minX, maxX, 30)
    yy = np.linspace(minY, maxY, 30)
    YY, XX = np.meshgrid(yy, xx)

    # XX.ravel(): 第一行，第二行, 拼成一行 [xx[0,0] xx[0,1] ... xx[1,0] xx[1,1] ...]
    # vstack 以后：
    # [[xx[0,0] xx[0,1] ... xx[1,0] xx[1,1] ...]
    #   yy[0,0] yy[0,1] ... yy[1,0] yy[1,1] ...]]
    # .T 以后
    # [[xx[0,0] yy[0,0]]
    #  [xx[0,1] yy[0,1]]
    #  ... ]
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax = plt.gca()
    ax.contour(
        XX,
        YY,
        Z,
        colors="k",
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=["--", "-", "--"],
    )
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.show()

    # proximity
    sub = addProximity(ts=c, k=20)
    s = sub.getSubsetByRatio(0.25, 0.25)
    plt.scatter(s.P[:, 0], s.P[:, 1], label="class 1")
    plt.scatter(s.N[:, 0], s.N[:, 1], label="class 0")

    X, y = s.toXY()
    clf = SVC(C=1, gamma=1)
    clf.fit(X, y)
    clf.fit(X, y)
    minX = -1
    maxX = 1
    minY = -1
    maxY = 1
    plt.xlim((minX - 0.2, maxX + 0.2))
    plt.ylim((minY - 0.2, maxY + 0.2))
    xx = np.linspace(minX, maxX, 30)
    yy = np.linspace(minY, maxY, 30)
    YY, XX = np.meshgrid(yy, xx)

    # XX.ravel(): 第一行，第二行, 拼成一行 [xx[0,0] xx[0,1] ... xx[1,0] xx[1,1] ...]
    # vstack 以后：
    # [[xx[0,0] xx[0,1] ... xx[1,0] xx[1,1] ...]
    #   yy[0,0] yy[0,1] ... yy[1,0] yy[1,1] ...]]
    # .T 以后
    # [[xx[0,0] yy[0,0]]
    #  [xx[0,1] yy[0,1]]
    #  ... ]
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax = plt.gca()
    ax.contour(
        XX,
        YY,
        Z,
        colors="k",
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=["--", "-", "--"],
    )
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.show()
