"""Insance selection methods."""

import logging
import os
import time

import numpy as np

import utils
import ismethod


def noselect(X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
    selected_idxs = np.arange(len(X))
    return selected_idxs

METHOD_FUNCS = [
    noselect,
    ismethod.lsh_ours,
    ismethod.lsh_dr,
    ismethod.lsh_bp,
    ismethod.pdoc_knn,
    ismethod.pdoc_rnn,
    ismethod.pdoc_gb,
    ismethod.ENNC,
    ismethod.NC,
    ismethod.PDOC_Voronoi,
    ismethod.LSH_IS_F_bs,
    ismethod.NE,
]

FUNC_NAMES_RETURN_XY = ["ENNC", "NC", "PDOC_Voronoi", "LSH_IS_F_bs", "NE"]


DICT_NAME_FUNC = {func.__name__: func for func in METHOD_FUNCS}


def _get_dict_name_func() -> dict:
    # dict_name_func = {func.__name__: func for func in METHOD_FUNCS}
    return DICT_NAME_FUNC


def get_funcname_kwargs(
    func_names: list[str] = None, handle_selrate=False
) -> dict[str, dict]:
    func_names = (
        func_names if func_names is not None else list(_get_dict_name_func().keys())
    )
    default_dict = _get_dict_name_func()

    dict_funcname_kwargs = {}
    for func_name in func_names:
        func = default_dict[func_name]
        param_default, param_required = utils.get_function_default_params(func)

        if handle_selrate:
            param_default["selection_rate"] = 0.1
        method_name = func_name
        dict_funcname_kwargs[method_name] = param_default
    return dict_funcname_kwargs


def sample_data(
    X: np.ndarray,
    y: np.ndarray,
    func_name: str,
    kwargs: dict = None,
    cache_dir: str = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Sample data from the original dataset.

    Args:
        X (np.ndarray): The original data.
        y (np.ndarray): The original label.
        func_name (str): The name of the instance selection method.
        kwargs (dict, optional): The kwargs of the selection method. Defaults to None.
        cache_dir (str, optional): The cache dir. If not None, the result will be cached. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray, float]: The sampled data, sampled label and the time of the method.
    """
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
            sub_idxs, time_method = utils.load(cache_path)
            logging.info("load from cache: %s", cache_path)
        else:
            time_start = time.process_time()  # record the cpu time
            sub_idxs = method_func(X, y, **kwargs)
            time_method = time.process_time() - time_start

            os.makedirs(cache_dir, exist_ok=True)
            # np.save(cache_path, sub_idxs)
            utils.save((sub_idxs, time_method), cache_path)
            logging.info("save to cache: %s", cache_path)
    else:
        time_start = time.process_time()  # record the cpu time
        sub_idxs = method_func(X, y, **kwargs)
        time_method = time.process_time() - time_start

    if len(sub_idxs) == 0:
        # empty array
        logging.warning(
            "sub_idxs is empty. will return empty array. X.shape=%s, func_name=%s, kwargs=%s",
            X.shape,
            func_name,
            kwargs,
        )
        X_sub = np.array([])
        y_sub = np.array([])
    else:
        # TODO method_func should all return selected_idxs for efficiency and consistency
        # but now some method_func return X_sub, y_sub
        # so we need to handle this
        if func_name in FUNC_NAMES_RETURN_XY:
            X_sub, y_sub = sub_idxs
        else:
            X_sub = X[sub_idxs]
            y_sub = y[sub_idxs]

    return X_sub, y_sub, time_method


if __name__ == "__main__":
    test_X_sub, test_y_sub, test_time_method = sample_data(
        X=np.random.rand(10, 3),
        y=np.array([1] * 6 + [-1] * 4),
        func_name="pdoc_knn",
        kwargs={
            "n_neighbors": 1,
            "thresholds": None,
            "selection_rate": 0.1,
            "selection_type": "far_random",
            "seed": 0,
        },
    )
    print(test_X_sub, test_y_sub, test_time_method)
