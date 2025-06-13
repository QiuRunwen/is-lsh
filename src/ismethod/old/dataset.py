# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:08:40 2020

@author: dhlsf
"""

import os
import random
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.datasets import make_blobs, make_circles
from sklearn.model_selection import train_test_split

from . import util


def split(X, y):
    """
    对原始的数据表格进行正负样本的划分
    Parameters
    ----------
    X : numpy.array/dataframe
        X[i]: 原始数中的第 i 个数据点
    y : numpy.array/dataframe
        y[i]: X[i]对应的标签，其中1为正样本，0为负样本

    Returns
    -------
    P : numpy.array
        P[i]表示第 i 个正样本点
    N : numpy.array
        N[i]表示第 i 个负样本点

    """
    # XyPair = list(zip(X, y))

    # P = [i[0] for i in XyPair if i[1] == 1]
    # N = [i[0] for i in XyPair if i[1] == 0 or i[1] == -1]

    # P = np.array(P)
    # N = np.array(N)

    # return P, N

    # modified by Qiu
    X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
    y = y.to_numpy() if isinstance(y, pd.Series) else y
    assert np.unique(y).size == 2

    tmp_filt = y == 1
    return X[tmp_filt], X[~tmp_filt]


def yTransform_reverse(y):
    if min(y) == -1:
        for i in range(len(y)):
            if y[i] == -1:
                y[i] = 0
    return y


class TrainSet:
    """
    根据P,N,PW和NW,决定每次选多少P和N，再重新返回类中作为初始化参数去拼接返回X和Y,作为训练数据
    """

    @ staticmethod
    def gen(X, y, dsDir=""):
        """
        把数据 X, 及标签 y 按照 y = 1 或 0 分成正负样本。用输入顺序作为一个样本点的 PW 及 NW

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        dsDir : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        P, N = split(X, y)
        # PW = np.array([-i for i in range(len(P))])  # 将正样本点输入的次序的负数作为PW
        # NW = np.array([-i for i in range(len(N))])  # 将负样本点输入的次序的负数作为NW
        
        # modified by Qiu
        PW = np.arange(0, -len(P), -1)
        NW = np.arange(0, -len(N), -1)
        
        return TrainSet(P, PW, N, NW, dsDir)

    def __init__(self, P, PW, N, NW, dsDir=""):
        if dsDir is not None:
            if dsDir:
                os.makedirs(dsDir, exist_ok=True)

        if not isinstance(P, np.ndarray):
            raise RuntimeError("P is not a np.array")

        if not isinstance(PW, np.ndarray):
            raise RuntimeError("PW is not a np.array")

        if not isinstance(N, np.ndarray):
            raise RuntimeError("N is not a np.array")

        if not isinstance(NW, np.ndarray):
            raise RuntimeError("NW is not a np.array")

        if len(P) != len(PW):
            raise RuntimeError(f"{len(P)=} != {len(PW)=}")
        if len(N) != len(NW):
            raise RuntimeError(f"{len(P)=} != {len(PW)=}")

        self.P = P
        self.PW = PW
        self.N = N
        self.NW = NW
        self.dsDir = dsDir

    def getName(self):
        if hasattr(self, "dsName"):
            return self.dsName
        return os.path.basename(self.dsDir)

    def createSubset(self, sP, sPW, sN, sNW, dsDir):
        return TrainSet(sP, sPW, sN, sNW, dsDir)

    def getSubset(self, countP, countN, returnIdx=False):
        # countP = min(countP, len(self.P))
        # countN = min(countN, len(self.N))

        # sPWindex = np.argpartition(self.PW, -countP)[-countP:]  # 用argpartition找出countP个最大的PW的下标
        # sNWindex = np.argpartition(self.NW, -countN)[-countN:]  # 用argpartition找出countN个最大的NW的下标

        # sP = np.array(self.P)[sPWindex]
        # sPW = np.array(self.PW)[sPWindex]

        # sN = np.array(self.N)[sNWindex]
        # sNW = np.array(self.NW)[sNWindex]

        if self.dsDir is None or self.dsDir == "":
            dsDir = None
        else:
            dsDir = os.path.join(self.dsDir, "pc-" + str(countP) + "-nc-" + str(countN))

        def getLargestByKey(key, value, count):
            if count < len(key):
                idx = np.argpartition(key, -count)[
                    -count:
                ]  # 用argpartition找出count个最大的key的下标
                k = np.array(key)[idx]
                v = np.array(value)[idx]
            else:
                idx = None
                k = np.array(key)
                v = np.array(value)

            return k, v, idx

        sPW, sP, sPWindex = getLargestByKey(self.PW, self.P, countP)
        sNW, sN, sNWindex = getLargestByKey(self.NW, self.N, countN)
        if returnIdx:
            return self.createSubset(sP, sPW, sN, sNW, dsDir), sPWindex, sNWindex
        else:
            return self.createSubset(sP, sPW, sN, sNW, dsDir)

    def getSubsetByRatio(self, pratio, nratio, returnIdx=False):
        countP = int(len(self.P) * pratio)
        countN = int(len(self.N) * nratio)

        return self.getSubset(countP, countN, returnIdx=returnIdx)

    def getSubsetByIndices(self, name, ascPIdx, ascNIdx):
        sP = self.P[ascPIdx]
        sPW = self.PW[ascPIdx]

        sN = self.N[ascNIdx]
        sNW = self.NW[ascNIdx]

        dsDir = os.path.join(self.dsDir, name)
        return self.createSubset(sP, sPW, sN, sNW, dsDir)

    def getSubsetBySampling(self, pnRatio, seed=0):
        rand = random.Random(seed)

        idxP = [i for i in range(len(self.P))]
        idxN = [i for i in range(len(self.N))]

        (pratio, nratio) = pnRatio
        selectedP = rand.sample(idxP, int(len(self.P) * pratio))
        selectedN = rand.sample(idxN, int(len(self.N) * nratio))

        return self.getSubsetByIndices(
            "pr={0},nr={1},seed={2}".format(pratio, nratio, seed), selectedP, selectedN
        )

    def toXY(self):
        TrainSetX = np.concatenate((self.P, self.N))
        TrainSetY = np.concatenate((np.ones(len(self.P)), np.zeros(len(self.N))))

        return TrainSetX, TrainSetY

    def splitTrainAndValidation(self, testRatio=0.2, random_state=0):
        """
        按照0.8：0.2的比例划分训练集和验证集
        """

        tP, vP, tPW, vPW = train_test_split(
            self.P, self.PW, test_size=testRatio, random_state=random_state
        )
        tN, vN, tNW, vNW = train_test_split(
            self.N, self.NW, test_size=testRatio, random_state=random_state
        )

        return TrainSet(
            tP,
            tPW,
            tN,
            tNW,
            self.dsDir + "-rand" + str(random_state) + "-tr" + str(testRatio) + "-ts",
        ), TrainSet(
            vP,
            vPW,
            vN,
            vNW,
            self.dsDir + "-rand" + str(random_state) + "-tr" + str(testRatio) + "-vs",
        )

    def plot_voronoi_cells_2D(self):
        X, y = self.toXY()
        v = Voronoi(X)
        voronoi_plot_2d(
            v, show_vertices=False, show_points=False
        )  # show_points = False就不会显示点

    def scatter2D(
        self, show_idx=False, pColor="#0000FF", nColor="#FF0000", s=150, edgecolors=None
    ):
        """
        把数据的前两维做散点图

        Parameters
        ----------
        pColor : 颜色, optional
            正类的颜色. 默认红色
        nColor : 颜色, optional
            负类的颜色. 默认蓝色
        s : numeric, optional
            点的大小. 默认150.

        Returns
        -------
        None.
        """
        plt.scatter(self.P[:, 0], self.P[:, 1], c=pColor, s=s, edgecolors=edgecolors)
        plt.scatter(self.N[:, 0], self.N[:, 1], c=nColor, s=s, edgecolors=edgecolors)
        if show_idx:
            for i in range(len(self.N)):
                plt.annotate(str(i), (self.N[i, 0], self.N[i, 1] + 0.1))
            for i in range(len(self.P)):
                plt.annotate(str(i + len(self.N)), (self.P[i, 0], self.P[i, 1] + 0.1))

    def plot_selected_2D(self, data, pColor="#0000FF", nColor="#FF0000", s=150):
        """
        原数据集的点为空心圆点；(本数据集）为筛选出来的数据集画成实心圆。

        Parameters
        ----------
        data : dataset.TrainSet
            原数据集
        pColor :
            正类的颜色，默认蓝色
        nColor :
            负类的颜色，默认红色
        s : numeric, optional
            点的大小. 默认150.

        Returns
        -------
        None.
        """
        plt.scatter(
            data.P[:, 0],
            data.P[:, 1],
            edgecolors=pColor,
            marker="o",
            facecolors="none",
            s=s,
        )
        plt.scatter(
            data.N[:, 0],
            data.N[:, 1],
            edgecolors=nColor,
            marker="o",
            facecolors="none",
            s=s,
        )

        self.scatter2D(pColor=pColor, nColor=nColor, s=s)

    # 下面两个函数是为了画图用
    def prepareForPlot(self, P, PW):  # 画图的时候才去sort
        P, PW = util.sortByKey(
            P, PW, asc=True, returnKey=True
        )  # 根据 PW 对 P 按照从小到大的顺序进行排序
        r = []
        for i in range(len(P)):
            r.append((P[i][0], P[i][1], i))
        return np.array(r)

    def plot2D(self, show_colorbar=False):
        """
        按照对象的权重 （越大表示离决策边界越近） 渐进着色，负类为紫色，正类为蓝色

        Returns
        -------
        None.

        """
        plt.style.use("ggplot")
        plt.rc("font", family="Times New Roman")

        p1 = plt.scatter(self.N[:, 0], self.N[:, 1], c=self.NW, cmap="RdPu")
        p2 = plt.scatter(self.P[:, 0], self.P[:, 1], c=self.PW, cmap="PuBu")
        if show_colorbar:
            plt.colorbar(p1, label="Negative")
            plt.colorbar(p2, label="Positive")
            plt.legend()

    def showWithTitle(self, title, show_colorbar=False):
        self.plot2D(show_colorbar=show_colorbar)
        plt.title(title)
        plt.show()

    def get_bounding_box(self, margin=0.5):
        """
        Compute a bounding box to contain all data points

        X_min[i] = min of P[:,i], N[:,i] - margin
        X_max[i] = max of P[:,i], N[:,i] + margin

        Parameters
        ----------
        margin : TYPE, optional
            DESCRIPTION. The default is 0.5.

        Returns
        -------
        x_min : TYPE
            DESCRIPTION.
        x_max : TYPE
            DESCRIPTION.

        """
        P_min = np.min(self.P, axis=0)
        N_min = np.min(self.N, axis=0)
        t = np.vstack((P_min, N_min))
        X_min = np.min(t, axis=0) - margin

        P_max = np.max(self.P, axis=0)
        N_max = np.max(self.N, axis=0)
        t = np.vstack((P_max, N_max))
        X_max = np.max(t, axis=0) + margin

        return X_min, X_max

    def set_xlim_ylim_2D(self, margin=0.5):
        X_min, X_max = self.get_bounding_box(margin)
        x_min, x_max = X_min[0], X_max[0]
        y_min, y_max = X_min[1], X_max[1]
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        return x_min, x_max, y_min, y_max

    def plot_decision_func(
        self, clf, margin=0.5, grid_gap=0.02, plot_contour=True, c="g", s=20
    ):
        assert self.P.shape[1] == 2

        # set bounding box and range for X- and Y-axis
        x_min, x_max, y_min, y_max = self.set_xlim_ylim_2D(margin)

        x_grids = np.arange(x_min, x_max, grid_gap)
        y_grids = np.arange(y_min, y_max, grid_gap)
        xx, yy = np.meshgrid(x_grids, y_grids)

        # Z[i,j] = f(x_grids[j], y_grids[i])
        # GBDT, SVM, LR, f((x,y)) 表示点 （x,y) 离决策面的距离, 正类 > 0, 负类 < 0
        # DT, f((x,y)) 表示点 (x,y) 是正类的概率， 正类 > 0.5, 负类 < 0.5
        temp = np.c_[xx.ravel(), yy.ravel()]
        if hasattr(clf, "decision_function"):  # GBDT,SVM,LR
            Z = clf.decision_function(temp)
            # boundary_value = 0
            boundary_value = 0.5
            Z = 1 / (1 + np.exp(-Z))
        else:
            Z = clf.predict_proba(temp)[:, 1]  # [:,j] 为数据点属于第 j 类的概率
            boundary_value = 0.5
        Z = Z.reshape(xx.shape)

        z_min = np.min(Z)
        z_max = np.max(Z)
        print("************* TODO: **************")
        print(f"{z_min = }")
        print(f"{z_max = }")

        # # draw contour
        if plot_contour:
            plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8, vmin=0, vmax=1)

        # for each y find the x that lie on boundary
        x_lst = []
        y_lst = []
        # 按每个 y 找 x 左右属于不同类的位置
        for i, y in enumerate(y_grids):  # 每行
            for j in range(len(x_grids) - 1):
                if Z[i, j] == boundary_value:
                    x_lst.append(x_grids[j])
                    y_lst.append(y)
                elif (Z[i, j] < boundary_value and Z[i, j + 1] > boundary_value) or (
                    Z[i, j] > boundary_value and Z[i, j + 1] < boundary_value
                ):
                    x_lst.append((x_grids[j] + x_grids[j + 1]) / 2)
                    y_lst.append(y)
        # 按每个 x 找 y 上下属于不同类的位置
        for j, x in enumerate(x_grids):  # 每行
            for i in range(len(y_grids) - 1):
                if Z[i, j] == boundary_value:
                    x_lst.append(x)
                    y_lst.append(y_grids[i])
                elif (Z[i, j] < boundary_value and Z[i + 1, j] > boundary_value) or (
                    Z[i, j] > boundary_value and Z[i + 1, j] < boundary_value
                ):
                    x_lst.append(x)
                    y_lst.append((y_grids[i] + y_grids[i + 1]) / 2)

        plt.scatter(x=x_lst, y=y_lst, c=c, s=s)


class TrainSetWithGT(TrainSet):  # grand truth
    # distP[i]: 是 P[i] 在决策边界距离 > 0 表示在对的一面
    # distN[i]: 是 N[i] 在决策边界距离 < 0 表示在对的一面
    def __init__(self, P, PW, distP, N, NW, distN, dsDir):
        super().__init__(P, PW, N, NW, dsDir)
        self.distP = distP
        self.distN = distN
        self.wrongSideCountP = np.sum(np.array(distP) <= 0)
        self.wrongSideCountN = np.sum(np.array(distN) >= 0)

    def plot2D(self, show_colorbar=False):
        super().plot2D(show_colorbar=show_colorbar)
        self.plotDecisionBoundary()

    def plotDecisionBoundary(self, c="g", linewidth=1.5):
        pass

    def is_positive_by_gt(self, X):
        """
        f[i] = True if X[i] is predicted as positive class by grand truth

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass

    def area_diff_ratio(self, clf, grid_gap=0.02):
        """
        用数值积分方式计算 GrandTruth 决策边界与 clf 决策边界之间的体积占数据集的特征空间体积的比

        Parameters
        ----------
        clf : TYPE
            DESCRIPTION.
        grid_gap : TYPE, optional
            DESCRIPTION. The default is 0.02.

        Returns
        -------
        area_diff_ratio : TYPE
            DESCRIPTION.

        """
        X_min, X_max = self.get_bounding_box(margin=0)
        X_range = (
            X_max - X_min
        )  # * 0.5, for two circle, this select only samples in the inner circle
        X_mid = (X_min + X_max) / 2
        X_min = X_mid - X_range / 2
        X_max = X_mid + X_range / 2
        x_grids = np.arange(X_min[0], X_max[0], grid_gap)
        y_grids = np.arange(X_min[1], X_max[1], grid_gap)
        xx, yy = np.meshgrid(x_grids, y_grids)

        # Z[i,j] = f(x_grids[j], y_grids[i])
        # GBDT, SVM, LR, f((x,y)) 表示点 （x,y) 离决策面的距离, 正类 > 0, 负类 < 0
        # DT, f((x,y)) 表示点 (x,y) 是正类的概率， 正类 > 0.5, 负类 < 0.5
        temp = np.c_[xx.ravel(), yy.ravel()]

        is_pos_gt = self.is_positive_by_gt(temp)
        is_pos_pred = clf.predict(temp) > 0  # 正样本变成 T, 负样本（-1 或者 0） 变成 F
        count = np.sum(is_pos_gt != is_pos_pred)
        area_diff_ratio = count / len(temp)
        return area_diff_ratio

    def showWrongClass(self, title):
        plt.style.use("ggplot")
        plt.rc("font", family="Times New Roman")

        pc = list(map(lambda x: "lightsteelblue" if x > 0 else "blue", self.distP))
        plt.scatter(
            self.P[:, 0], self.P[:, 1], c=pc
        )  # ,cmap =cm_bright,edgecolors='k')
        self.plotDecisionBoundary()

        plt.title(title + " Positive")
        plt.legend()
        plt.show()

        nc = list(map(lambda x: "plum" if x < 0 else "purple", self.distN))
        plt.scatter(
            self.N[:, 0], self.N[:, 1], c=nc
        )  # ,cmap =cm_bright,edgecolors='k')
        self.plotDecisionBoundary()
        plt.title(title + " Negative")
        plt.legend()
        plt.show()


class TrainSet2Guasian(TrainSetWithGT):
    def gen(
        dsDir="../test/two-guasian",
        n_samples=[500, 500],
        cluster_std=[0.7, 0.7],
        random_state=0,
    ):
        """
        Parameters
        ----------
        dsDir : str
            2Guasian数据集存的位置. The default is "../test/two-guasian".
        n_samples : int
            正负样本选择的比例. The default is [500,500].
        random_state : int, optional
            随机种子数. The default is 0.

        Returns
        -------
        class.

        """

        X, y = make_blobs(
            n_samples=n_samples,
            n_features=2,
            centers=[[-1, 0], [1, 0]],
            cluster_std=cluster_std,
            random_state=random_state,
        )
        P, N = split(X, y)

        return TrainSet2Guasian(dsDir, P, N)

    def __init__(self, dsDir, P, N):
        PW = -abs(P[:, 0])  # 正类点离边界的真实距离的负数作为PW(越大，越离边界近)
        NW = -abs(N[:, 0])  # 负类点离边界的真实距离的负数作为NW(越大，越离边界近)
        distP = np.copy(P[:, 0])
        distN = np.copy(N[:, 0])
        super().__init__(P, PW, distP, N, NW, distN, dsDir)

    def plotDecisionBoundary(self, c="g", linewidth=1.5):
        minY = min(min(self.P[:, 1]), min(self.N[:, 1]))
        maxY = max(max(self.P[:, 1]), max(self.N[:, 1]))
        y = np.linspace(minY, maxY, num=100)
        plt.plot(y * 0, y, c=c, linewidth=linewidth)

    def is_positive_by_gt(self, X):
        return X[:, 0] > 0

    # def plot2D(self):
    #     super().plot2D()
    #     self()
    #     minY = min(min(self.P[:,1]), min(self.N[:,1]))
    #     maxY = max(max(self.P[:,1]), max(self.N[:,1]))
    #     y = np.linspace(minY, maxY, num=100)
    #     plt.plot(y*0,y,color="g")

    def splitTrainAndValidation(self, testRatio=0.2, random_state=0):
        """
        按照0.8：0.2的比例划分训练集和验证集
        """

        tP, vP, tPW, vPW = train_test_split(
            self.P, self.PW, test_size=testRatio, random_state=random_state
        )
        tN, vN, tNW, vNW = train_test_split(
            self.N, self.NW, test_size=testRatio, random_state=random_state
        )

        return TrainSet2Guasian(self.dsDir + "-ts", tP, tN), TrainSet2Guasian(
            self.dsDir + "-vs", vP, vN
        )

    def createSubset(self, sP, sPW, sN, sNW, dsDir):
        return TrainSet2Guasian(dsDir, sP, sN)


class TrainSet2Circle(TrainSetWithGT):
    def gen(dsDir="../test/two-circles", n_samples=1000, noise=0.2, random_state=1):
        inner_r = 0.5
        X, y = make_circles(
            n_samples=n_samples, noise=noise, factor=inner_r, random_state=random_state
        )
        dsDir = dsDir + "-rand" + str(random_state)
        P, N = split(X, y)

        return TrainSet2Circle(dsDir, P, N, 0.75)

    def __init__(self, dsDir, P, N, R):
        self.R = R  # decision boundary

        def dist2circle(P, r):
            x1 = P[:, 0]
            x2 = P[:, 1]
            return r - np.sqrt(x1 * x1 + x2 * x2)

        distP = dist2circle(P, self.R)
        distN = dist2circle(N, self.R)

        PW = -np.abs(distP)
        NW = -np.abs(distN)

        super().__init__(P, PW, distP, N, NW, distN, dsDir)

    def plotDecisionBoundary(self, c="g", linewidth=1.5):
        circle1 = plt.Circle((0, 0), self.R, color=c, fill=False, linewidth=linewidth)
        plt.gcf().gca().add_artist(circle1)

    def is_positive_by_gt(self, X):
        return np.sqrt(X[:, 0] * X[:, 0] + X[:, 1] * X[:, 1]) < self.R

    def splitTrainAndValidation(self, testRatio=0.2, random_state=0):
        """
        按照0.8：0.2的比例划分训练集和验证集
        """

        tP, vP, tPW, vPW = train_test_split(
            self.P, self.PW, test_size=testRatio, random_state=random_state
        )
        tN, vN, tNW, vNW = train_test_split(
            self.N, self.NW, test_size=testRatio, random_state=random_state
        )

        return TrainSet2Circle(self.dsDir + "-ts", tP, tN, self.R), TrainSet2Circle(
            self.dsDir + "-vs", vP, vN, self.R
        )

    def createSubset(self, sP, sPW, sN, sNW, dsDir):
        return TrainSet2Circle(dsDir, sP, sN, self.R)


class TrainSetXOR(TrainSetWithGT):
    def gen(dsDir="../test/XOR", n_samples=250, noise=0.8, random_state=0):
        np.random.seed(random_state)

        def gen_clusters(cx1, cx2):
            mean = [cx1, cx2]
            cov = [[noise * noise, 0], [0, noise * noise]]
            data = np.random.multivariate_normal(mean, cov, n_samples)
            return data

        data_p1 = gen_clusters(1, 1)
        data_p2 = gen_clusters(-1, -1)
        data_n1 = gen_clusters(1, -1)
        data_n2 = gen_clusters(-1, 1)

        P = np.vstack((data_p1, data_p2))
        N = np.vstack((data_n1, data_n2))

        return TrainSetXOR(dsDir, P, N)

    def __init__(self, dsDir, P, N):
        PW = -np.min(np.abs(P), axis=1)  # min {|x1|, |x2|} 是点到决策边界的距离
        NW = -np.min(np.abs(P), axis=1)
        distP = P[:, 0] * P[:, 1]
        distN = N[:, 0] * N[:, 1]
        super().__init__(P, PW, distP, N, NW, distN, dsDir)

    def plotDecisionBoundary(self, c="g", linewidth=1.5):
        minX = min(min(self.P[:, 0]), min(self.N[:, 0]))
        maxX = max(max(self.P[:, 0]), max(self.N[:, 0]))
        x = np.linspace(minX, maxX, num=100)
        plt.plot(x, x * 0, c=c, linewidth=linewidth)

        minY = min(min(self.P[:, 1]), min(self.N[:, 1]))
        maxY = max(max(self.P[:, 1]), max(self.N[:, 1]))
        y = np.linspace(minY, maxY, num=100)
        plt.plot(y * 0, y, c=c, linewidth=linewidth)

    def is_positive_by_gt(self, X):
        return X[:, 0] * X[:, 1] > 0  # 1,3象限是正样本，2，4象限是负样本


def readData(filePath):
    """
    从filePath目录下的 X.csv 及 y.csv 分别读取数据矩阵 X 及标签 y
    每个数据点用它在输入中的顺序作为PW、NW

    Parameters
    ----------
    filePath :
        需要读取的数据文件的位置.

    Returns
    -------
    X : numpy.array
        原始未去噪的训练集X.
    y : numpy.array
        原始未去噪的训练集Y(1D array).
    """

    X = pd.read_csv(os.path.join(filePath, "X.csv"))
    X = X.values
    y = pd.read_csv(os.path.join(filePath, "y.csv"))
    y = y.values
    y = y.reshape((-1,))

    return TrainSet.gen(X, y, filePath)


def readXy(filePath):
    """
    从filePath目录下的 X.npy 及 y.npy 分别读取数据矩阵 X 及标签 y
    每个数据点用它在输入中的顺序作为PW、NW

    Parameters
    ----------
    filePath : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    fileX = os.path.join(filePath, "X.npy")
    X = np.load(fileX, allow_pickle=True)
    filey = os.path.join(filePath, "y.npy")
    y = np.load(filey)
    return TrainSet.gen(X, y, filePath)


def loadTwoGuasianNoise(n_samples=[500, 500], noise=0.7, seed=0):
    dsDir = "../test/two-guasian{0}-seed={1}-n_samples{2}".format(
        noise, seed, n_samples
    )
    return TrainSet2Guasian.gen(
        dsDir=dsDir, n_samples=n_samples, cluster_std=[noise, noise], random_state=seed
    )


def loadTwoCircleNoise(noise=0.2, seed=1, n_samples=1000):
    dsDir = "../test/two-circiles_{0}noise-seed={1}".format(noise, seed)
    return TrainSet2Circle.gen(
        dsDir=dsDir, n_samples=n_samples, noise=noise, random_state=seed
    )


def loadXOR(seed=0, sigma=0.5, Eachn_samples=250):
    dsDir = "../test/XOR_{0}sigma-seed={1}".format(sigma, seed)
    return TrainSetXOR.gen(
        dsDir=dsDir, random_state=seed, sigma=sigma, Eachn_samples=Eachn_samples
    )


def loadIMDB(dataset="TrainSet"):
    return readXy("../NewDataSet/IMDB/" + dataset)


def gen_voronoi(dsDir="../result/sample_for_voronoi"):
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
            [2.2, 4.3],
            [3, 5.8],
            [4.2, 4.9],
            [4.8, 2.8],
            [5.5, 2.5],
            [6.5, 3.4],
            [6.8, 4.4],
            [7.3, 4.5],
            [5.5, 5],
            [5.3, 3.8],
            [3.2, 4.2],
            [3.5, 6],
            [2.8, 3.8],
            [3.8, 2],
            [0.9, 7],
        ]
    )
    y = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
        ]
    )
    return TrainSet.gen(X, y, dsDir)


###############################################################################
# Usage:


def testDraw2Guasion():
    a = TrainSet2Guasian()
    a.plot2D()
    plt.show()


def testDraw2Circle():
    a = TrainSet2Circle()
    a.plot2D()
    plt.show()


###############################################################################
# Unit Testing


class TestDataSet(unittest.TestCase):
    def test_split01(self):
        X = [1, 2, 3, 4, 5]
        y = [0, 1, 1, 0, 1]
        P, N = split(X, y)
        np.testing.assert_array_equal(P, [2, 3, 5])
        np.testing.assert_array_equal(N, [1, 4])


def expTwoGuassian1wPlot():
    twoGuasianNoise1w = loadTwoGuasianNoise(n_samples=[5000, 5000], noise=1.5)
    subsetTwoGuassianNoise1w = twoGuasianNoise1w.getSubsetByRatio(0.2, 0.2)
    print("=========", subsetTwoGuassianNoise1w.dsDir)
    subsetTwoGuassianNoise1w.showWithTitle("subsetTwoGuassianNoise1w")

    twoGuasianNoise1w.showWithTitle("2G1.5_1W")
    twoGuasianNoise1w.showWrongClass("2G0.7")
    print(twoGuasianNoise1w.wrongSideCountP, twoGuasianNoise1w.wrongSideCountN)


if __name__ == "__main__":
    guas = loadTwoGuasianNoise(n_samples=[50, 50])

    # guas.showWithTitle("2G0.7")
    # guas.showWrongClass("2G0.7")
    # print(guas.wrongSideCountP, guas.wrongSideCountN)

    # twoCircle02 = loadTwoCircleNoise()
    # twoCircle02.showWithTitle("2C0.2")
    # twoCircle02.showWrongClass("2C0.2")

    # twoCircle01 = loadTwoCircleNoise(noise = 0.1)
    # twoCircle01.showWithTitle("2C0.1")
    # twoCircle01.showWrongClass("2C0.1")
    # expTwoGuassian1wPlot()

    # from playsound import playsound
    # playsound('../Canon.mp3')

    import matplotlib.pyplot as plt
    from mlxtend.plotting import plot_decision_regions

    plt.style.use("ggplot")
    plt.rc("font", family="Times New Roman")

    import os

    from sklearn.svm import SVC

    xor = loadXOR(seed=0, sigma=0.8, Eachn_samples=250)
    print(xor.dsDir)
    xor.showWithTitle("XOR")
    # TrainSetX, TrainSetY = xor.toXY()
    # TrainSetY = TrainSetY.astype(int)
    # import time
    # startTime = time.time()
    # clf = SVC(C = 10, probability = True, random_state = 0)
    # clf.fit(TrainSetX, TrainSetY)

    # if TrainSetX.shape[1] == 2:
    #     plot_decision_regions(TrainSetX, TrainSetY.astype(np.integer), clf, legend=2)
    #     plt.show()
    # y_predict = clf.predict(TrainSetX)

    # from sklearn.metrics import accuracy_score
    # acc = accuracy_score(y_predict,TrainSetY)
