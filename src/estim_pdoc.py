"""Estimate the pdoc using different methods based on neighbors."""

import os
import logging
import time
import numpy as np
from ismethod import pdoc, lshqiu
import utils


def k_nn(X, y, *, n_neighbors=30, use_weight=False):
    ls_neighbor_idxs, ls_neighbor_weights = pdoc.get_neighbor_idx_and_weight(
        X=X,
        n_neighbors=n_neighbors,
    )
    if not use_weight:
        ls_neighbor_weights = None
    pdoc_estim = pdoc.estimate_proba_diff_oppo_cls(
        y=y,
        ls_neighbor_idxs=ls_neighbor_idxs,
        ls_neighbor_weights=ls_neighbor_weights,
        use_fast=True,
    )
    return pdoc_estim


def r_nn(X, y, *, radius=0.1, use_weight=False):
    ls_neighbor_idxs, ls_neighbor_weights = pdoc.get_neighbor_idx_and_weight(
        X=X, radius=radius
    )
    if not use_weight:
        ls_neighbor_weights = None

    pdoc_estim = pdoc.estimate_proba_diff_oppo_cls(
        y=y,
        ls_neighbor_idxs=ls_neighbor_idxs,
        ls_neighbor_weights=ls_neighbor_weights,
        use_fast=True,
    )
    return pdoc_estim


def lsh(X, y, *, w=1, t=5, L=10, use_weight=True, seed=None):
    ls_neighbor_idxs, ls_neighbor_weights = lshqiu.get_neighbor_idx_and_weight(
        X, w=w, t=t, L=L, seed=seed
    )
    if not use_weight:
        ls_neighbor_weights = None

    pdoc_estim = pdoc.estimate_proba_diff_oppo_cls(
        y=y,
        ls_neighbor_idxs=ls_neighbor_idxs,
        ls_neighbor_weights=ls_neighbor_weights,
        use_fast=True,
    )
    return pdoc_estim


METHOD_FUNCS = [
    k_nn,
    r_nn,
    lsh,
]


def _get_dict_name_func() -> dict:
    dict_name_func = {func.__name__: func for func in METHOD_FUNCS}
    return dict_name_func


def estimate_pdoc(
    X: np.ndarray,
    y: np.ndarray,
    func_name: str,
    kwargs: dict = None,
    cache_dir: str = None,
):
    dict_name_func = _get_dict_name_func()
    method_func = dict_name_func[func_name]
    kwargs = kwargs if kwargs is not None else {}
    if cache_dir is not None:
        cache_name = func_name + "-"
        for key, val in kwargs.items():
            cache_name += f"{key}={val}_"
        cache_name = f"{hash(cache_name[:-1])}.pkl"
        cache_path = os.path.join(cache_dir, cache_name)
        if os.path.exists(cache_path):
            # sub_idxs = np.load(cache_path)
            pdoc_estim, time_method = utils.load(cache_path)
            logging.info("load from cache: %s", cache_path)
        else:
            time_start = time.process_time()  # record the cpu time
            pdoc_estim = method_func(X, y, **kwargs)
            time_method = time.process_time() - time_start

            os.makedirs(cache_dir, exist_ok=True)
            # np.save(cache_path, sub_idxs)
            utils.save((pdoc_estim, time_method), cache_path)
            logging.info("save to cache: %s", cache_path)
    else:
        time_start = time.process_time()  # record the cpu time
        pdoc_estim = method_func(X, y, **kwargs)
        time_method = time.process_time() - time_start

    return pdoc_estim, time_method
