# -*- coding: utf-8 -*-
"""
放通用的函数，其他文件使用时，用import进行调用。
"""
import os
import random
import time
from functools import wraps

import numpy as np
from scipy.spatial import distance


def setSeed(seed_value):
    """
    Parameters
    ----------
    seed_value : int
        随机种子数，以便程序可以重复调用.
    Returns
    -------
    None.
    """

    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # 3. Set the `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    # tf.random.set_seed(seed_value)

    # 5. Configure a new global `tensorflow` session    # tf.__version__: 2.2.0
    # session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    # sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    # tf.compat.v1.keras.backend.set_session(sess)


def run_time(func):
    """
    函数装饰器，用来记录函数运行时间
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.process_time()
        result = func(*args, **kwargs)
        end = time.process_time()
        cost_time = end - start
        # print("{} is running".format(func.__name__))
        # print("func {} run time {}".format(func.__name__, cost_time))
        return result

    return wrapper


def computeOrLoad(file, func):
    if not file.endswith(".npy"):
        raise RuntimeError("file: {0} not ends with .npy".format(file))

    if os.path.exists(file):
        # print("load from file: " + file)
        return np.load(file, allow_pickle=True)

    else:
        start = time.process_time()
        a = func()
        runTime = time.process_time() - start
        # os.makedirs(os.path.dirname(file), exist_ok=True)
        # np.save(file, a)
        # np.save(file[:-4] + "-time.npy", runTime)
        # print("save to file: " + file, file[:-4] + "-time.npy")
        return a


def rank(a, asc=True):
    """
    为a中的每个元素计算它的排名
    返回r: r[i] = j，表示a[i]的排名为j
    asc： True: 从小到大排序； False 从大到小排序
    a：是list或者numpy.array
    """
    n = len(a)
    b = np.argsort(a)
    r = np.zeros(n, dtype=int)
    if asc:
        for i in range(n):
            r[b[i]] = i

    else:
        last = n - 1
        for i in range(n):
            r[b[i]] = last - i

    return r


def sortByKey(a, key, asc=True, returnKey=False):
    """
    按照key从小到大对a进行排序，返回排完序结果；
    如果需要从大到小排，asc=False
    a，key可以是python list或者numpy.array
    """
    if not isinstance(key, np.ndarray):
        key = np.array(key)

    if asc:
        b = np.argsort(key)
    else:
        b = np.argsort(-key)

    if not isinstance(a, np.ndarray):
        a = np.array(a)

    if returnKey:
        return a[b], key[b]
    else:
        return a[b]


def cumulatedCorrectRank(d, proximity):
    """
    计算top k个 proximity 点中有多少个点是真的离边界近的 top k 个点

    Parameters
    ----------
    d : python list or numpy.array
        每一个点离边界的真实距离，d越小，表示点离边界越近.
    proximity : python list or numpy.array
        通过byChain/byFrequency计算出来的每个点离边界远近程序，proximity越大，说明点离边界越近.

    Returns
    -------
    g : numpy.array
        按照proximity排最大的k个它们有多少真实排名<=k.
    """

    n = len(d)
    dRank = rank(d, asc=True)
    r = sortByKey(dRank, proximity, asc=False)

    g = np.zeros(n, dtype=int)
    count = 0
    t = np.zeros(n, dtype=int)

    for k in range(n):
        rk = r[k]
        newC = t[k]

        if rk <= k:
            newC = newC + 1
        else:
            t[rk] = 1

        g[k] = count + newC

        count = g[k]
    return g


def auCCR(g):
    """
    Normalized area uder cumulated correct rank
    Perfect score is 1.0

    Parameters
    ----------
    g : int array
        computed by cumulatedCorrectRank

    Returns
    -------
    real number in [0,1]
    """
    n = len(g)
    perfect = n * (n + 1) / 2
    return sum(g) / perfect


###############################################################################
# CTD: cumulated true distance, proximity 排序与真实排序越接近，累计真实距离曲线
#      下面的面积越小
###############################################################################
def cumulatedSum(r):
    """
    Parameters
    ----------
    r : numeric array

    Returns
    -------
    g : numeric array
        g[i] = sum(r[0:i])
    """
    g = np.zeros(len(r))
    g[0] = r[0]
    for i in range(1, len(r)):
        g[i] = r[i] + g[i - 1]
    return g


def cumulatedTrueDist(d, proximity):
    """
    Parameters
    ----------
    d : python list or numpy.array
        每一个点离边界的真实距离，d越小，表示点离边界越近.
    proximity : python list or numpy.array
        通过byChain/byFrequency计算出来的每个点离边界远近程序，proximity越大，说明点离边界越近.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    r = sortByKey(d, proximity, asc=False)
    return cumulatedSum(r)


def perfectCTD(d):
    """
    完美CTD曲线，即按点到决策边界的真实距离升序排
    Parameters
    ----------
    d : python list or numpy.array
        每一个点离边界的真实距离，d越小，表示点离边界越近.
    """
    g = cumulatedSum(sorted(d))
    return sum(g)


def auCTD(g, perfectCTD):
    return sum(g) / perfectCTD


def randomAvgDist(P, k, seed=1, dist=distance.euclidean):
    """
    从P中随机选k个点，计算它们之间的平均距离
    Parameters
    ----------
    P : P[i]表示第i个数据点.
    k : int
        随机挑选的个数.
    seed : TYPE, optional
        DESCRIPTION. The default is 1.
    dist : TYPE, optional
        DESCRIPTION. The default is distance.euclidean.

    """

    if type(k) is not int:
        raise RuntimeError("type(k): {0} is not int, k = {1}".format(type(k), k))

    plen = len(P)
    if plen <= 1:
        return 0

    np.random.seed(seed)
    s = 0
    for i in range(k):
        i1 = np.random.randint(plen, size=1)
        i2 = i1
        while i2 == i1:
            i2 = np.random.randint(plen, size=1)
        s = s + dist(P[i1], P[i2])
    return s / k


###############################################################################

import unittest


class TestUtil(unittest.TestCase):
    def test_rank01(self):
        a = [0.8, 0.7, 0.9]
        r = [1, 2, 0]
        b = rank(a, asc=False)
        np.testing.assert_array_equal(b, r)

    def test_rank02(self):
        a = [0.3, 0.5, 0.7, 0.6, 0.4]
        r = [0, 2, 4, 3, 1]
        b = rank(a, asc=True)
        np.testing.assert_array_equal(b, r)

    def test_rank03(self):
        a = [0.3, 0.5, 0.7, 0.6, 0.4]
        r = [4, 2, 0, 1, 3]
        b = rank(a, asc=False)
        np.testing.assert_array_equal(b, r)

    def test_sortByKey01(self):
        a = [3, 1, 2]
        k = [3.5, 2.6, 7]
        np.testing.assert_array_equal(sortByKey(a, k), [1, 3, 2])

    def test_sortByKey02(self):
        a = [3, 1, 2]
        k = [3.5, 2.6, 7]
        np.testing.assert_array_equal(sortByKey(a, k, False), [2, 3, 1])

    def test_sortByKey03(self):
        k = [3.5, 2.6, 7]
        np.testing.assert_array_equal(sortByKey(k, k), [2.6, 3.5, 7])

    def test_sortByKey04(self):
        k = [3.5, 2.6, 7]
        np.testing.assert_array_equal(sortByKey(k, k, False), [7, 3.5, 2.6])

    def test_sortByKey05(self):
        a = [3, 1, 2]
        k = [3.5, 2.6, 7]
        sa, sk = sortByKey(a, k, returnKey=True)
        np.testing.assert_array_equal(sa, [1, 3, 2])
        np.testing.assert_array_equal(sk, [2.6, 3.5, 7])

    def test_sortByKey06(self):
        a = [3, 1, 2]
        k = [3.5, 2.6, 7]
        sa, sk = sortByKey(a, k, asc=False, returnKey=True)
        np.testing.assert_array_equal(sa, [2, 3, 1])
        np.testing.assert_array_equal(sk, [7, 3.5, 2.6])

    def test_cumulatedCorrectRank01(self):
        d = [5, 4.3, 7]
        p = [0.9, 0.6, 0.8]
        g = cumulatedCorrectRank(d, p)
        np.testing.assert_array_equal(g, [0, 1, 3])

    def test_cumulatedCorrectRank02(self):
        d = [5, 4.3, 7, 9]
        p = [0.9, 0.6, 0.8, 0.5]
        g = cumulatedCorrectRank(d, p)
        np.testing.assert_array_equal(g, [0, 1, 3, 4])

    def test_auCCR01(self):
        d = [5, 4.3, 7]
        p = [0.9, 0.6, 0.8]
        g = cumulatedCorrectRank(d, p)
        a = auCCR(g)
        self.assertAlmostEqual(a, 4 / 6)

    def test_auCCR02(self):
        d = [5, 4.3, 7, 9]
        p = [0.9, 0.6, 0.8, 0.5]
        g = cumulatedCorrectRank(d, p)
        a = auCCR(g)
        self.assertAlmostEqual(a, 8 / 10)

    def test_cumulatedSum01(self):
        r = [3, 2, 4]
        g = cumulatedSum(r)
        np.testing.assert_array_equal(g, [3, 5, 9])
        p = [0.7, 0.3, 0.9]
        g = cumulatedTrueDist(r, p)  # [4, 3, 2]
        np.testing.assert_array_equal(g, [4, 7, 9])
        sCTD = perfectCTD(r)  # [2, 3, 4] => [2, 5, 9] => 16
        self.assertEqual(sCTD, 16)
        a = auCTD(g, sCTD)  # [4, 3, 2] => [4, 7, 9] => 20
        self.assertAlmostEqual(a, 20 / 16)


def testComputeOrLoad():
    file = "../test/util-computeOrLoad01.npy"

    def f():
        return [2, 1, 3]

    a = computeOrLoad(file, f)
    print(a)

    a = computeOrLoad(file, f)
    print(a)


if __name__ == "__main__":
    unittest.main()

#    P = np.array([1, 2, 3, 4, 5])
#    d = randomAvgDist(P, 10000)
#    print(d)
