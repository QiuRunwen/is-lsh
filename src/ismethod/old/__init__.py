"""
Module for the old instance selection methods.
Most codes are from https://github.com/ProfZHUWB/PDOC-V.
Reorganized by QiuRunwen.
"""

import numpy as np

from .exp import byEntropy20, byAC, byLSH_IS_F_bs, byVoronoiFilter02, byFreq
from .dataset import TrainSet


def format_ysub(y: np.ndarray, y_sub: np.ndarray):
    """If y unique is [0, 1], then y_sub should be [0, 1].
    If y unique is [-1, 1], then y_sub should be [-1, 1].
    """
    y_unique = np.unique(y)  # [-1, 1] or [0, 1]. np.unique will sort the array.

    y_sub_new = np.where(y_sub == 1, y_unique[1], y_unique[0])

    return y_sub_new


def ENNC(
    X: np.ndarray, y: np.ndarray, *, selection_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Zhu F, Yang J, Gao J, et al. Extended nearest neighbor chain induced instance-weights for SVMs[J].
    Pattern Recognition, 2016, 60:863-874.
    """
    pos_count = np.sum(y == 1)
    neg_count = X.shape[0] - pos_count

    p_count = int(pos_count * selection_rate)
    n_count = int(neg_count * selection_rate)

    ts = byAC(
        TrainSet.gen(X, y),
        p_count=p_count,
        n_count=n_count,
    )
    X_sub, y_sub = ts.toXY()
    y_sub = format_ysub(y, y_sub)

    return X_sub, y_sub


def NC(
    X: np.ndarray, y: np.ndarray, *, selection_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Zhu Z, Wang Z, Li D, et al. NearCount: Selecting critical instances based on the cited
    counts of nearest neighbors[J]. Knowledge-Based Systems, 2020, 190:105196.
    """
    pos_count = np.sum(y == 1)
    neg_count = X.shape[0] - pos_count

    p_count = int(pos_count * selection_rate)
    n_count = int(neg_count * selection_rate)

    ts = byFreq(
        TrainSet.gen(X, y),
        p_count=p_count,
        n_count=n_count,
    )
    X_sub, y_sub = ts.toXY()
    y_sub = format_ysub(y, y_sub)

    return X_sub, y_sub


def PDOC_Voronoi(
    X: np.ndarray, y: np.ndarray, *, selection_rate: float, seed: int = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Y. Fu, K. Liu, and W. Zhu, “Instance Selection Via Voronoi Neighbors for Binary Classification Tasks,”
    IEEE Transactions on Knowledge and Data Engineering, pp. 113, 2023, doi: 10.1109/TKDE.2023.3328952.
    """
    pos_count = np.sum(y == 1)
    neg_count = X.shape[0] - pos_count

    p_count = int(pos_count * selection_rate)
    n_count = int(neg_count * selection_rate)
    if seed is not None:
        np.random.seed(seed)

    ts = byVoronoiFilter02(
        TrainSet.gen(X, y),
        p_count=p_count,
        n_count=n_count,
    )
    X_sub, y_sub = ts.toXY()
    y_sub = format_ysub(y, y_sub)

    return X_sub, y_sub


def LSH_IS_F_bs(
    X: np.ndarray, y: np.ndarray, *, selection_rate: float, seed: int = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Á. Arnaiz-González, J.-F. Díez-Pastor, J. J. Rodríguez, and C. García-Osorio,
    “Instance selection of linear complexity for big data,” Knowledge-Based Systems,
    vol. 107, pp. 83-95, Sep. 2016, doi: 10.1016/j.knosys.2016.05.056.
    """
    if seed is not None:
        np.random.seed(seed)

    ts = byLSH_IS_F_bs(
        TrainSet.gen(X, y), p_count=None, n_count=None, w_alpha=selection_rate
    )
    X_sub, y_sub = ts.toXY()
    y_sub = format_ysub(y, y_sub)

    return X_sub, y_sub


def NE(X: np.ndarray, y: np.ndarray, *, selection_rate: float, seed: int = None):
    """
    H. Shin and S. Cho, “Neighborhood Property–Based Pattern Selection for Support Vector Machines,”
    Neural Computation, vol. 19, no. 3, pp. 816–855, Mar. 2007, doi: 10.1162/neco.2007.19.3.816.
    """
    pos_count = np.sum(y == 1)
    neg_count = X.shape[0] - pos_count

    p_count = int(pos_count * selection_rate)
    n_count = int(neg_count * selection_rate)
    if seed is not None:
        np.random.seed(seed)

    ts = byEntropy20(
        TrainSet.gen(X, y),
        p_count=p_count,
        n_count=n_count,
    )
    X_sub, y_sub = ts.toXY()
    y_sub = format_ysub(y, y_sub)

    return X_sub, y_sub


__all__ = ["ENNC", "NC", "PDOC_Voronoi", "LSH_IS_F_bs", "NE"]
