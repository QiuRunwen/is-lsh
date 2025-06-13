"""Implement
1. p-stable LSH
2. Our instance selection method based on LSH
@author: QiuRunwen
"""

# import sys
import itertools
import json
import logging
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate, optimize, special
from sklearn.neighbors import NearestNeighbors

# add exeutable path `../../Library` of glpk and ipopt to PATH
# visible in this process + all children
curr_file_path = Path(__file__)
os.environ["PYOMO_CONFIG_DIR"] = os.path.abspath(
    os.path.join(curr_file_path.parent.parent.parent.resolve(), "Library")
)

import pyomo.environ as pyo


# from pyomo.common import Executable
# print(os.environ["PYOMO_CONFIG_DIR"])
# print(Executable("ipopt").path())

if __name__ == "__main__":
    import pdoc
    import utils
else:
    from . import pdoc
    from . import utils


def lsh_2stable(X: np.ndarray, w: float = 1, t: int = 1, L: int = 1, seed: int = None):
    """2-stable Local sensitive hash.

    Args:
        X (np.ndarray): Datasets with shape (size, dimension)
        w (float, optional): parameter of pstable-LSH. Defaults to 1.
        t (int, optional): Times of lsh. g(x)=(h_1(x),...,h_t(x)). Defaults to 1.
        L (int, optional): Number of hash tables. Defaults to 1.
        seed (int, optional): Random seed to reproduce results.

    Returns:
        List[np.ndaaray]: [HashTable1, HashTable2,..,HashTableL].
    """
    if w <= 0 or t <= 0 or L <= 0:
        raise ValueError(
            f"`w`, `t`, `L` should be all greater than 0. Now they are {w,t,L}"
        )

    # TODO np.random.default_rng and np.random.RandomState are not controlled by seed_everything
    if seed is None:
        # current implementation is not thread-safe when seed is None
        seed = np.random.get_state()[1][0]  # get the seed of the global random state
    rng = np.random.default_rng(seed)
    n, d = X.shape

    # TODO (1) no considering new points (2) list implementation can be faster
    # (1) In future, save `np.random.randn(d,t)`和`np.random.random(t)` for new point
    # (2) Use matrix multiplication to replace the for loop
    return [
        np.floor((X @ rng.standard_normal((d, t)) / w + rng.uniform(0, 1, t))).astype(
            int
        )
        for _ in range(L)
    ]
    # return np.floor((X @ rng.standard_normal((d,t, L)) / w + rng.uniform(0,1, (t,L)))).astype(int)


def convert_hashtable(
    hashtables: list[np.ndarray],
) -> tuple[list[dict], list[list[set]]]:
    """Convert hashtable to `hashvalue_i:bucket_i` like `{g_1: [pIndex1,pIndexk,...], g_2:[pIndex2,...],... }`

    Args:
        hashtables (list[np.ndarray]): _description_

    Returns:
        tuple[list[dict], list[list[set]]]:
        ls_dict : `[{g_1: [pIndex1,pIndexk,...], g_2:[pIndex2,...],... }, {...},... ]`
        ls_buckets: `[ [[pIndex1,pIndexk,...],[pIndex2,pIndexk,...],...],  [...],...  ]`
    """

    # TODO 进一步加速
    # L = len(hashtables)  # 表的个数
    n = len(hashtables[0])  # 数据集的size

    def map_func(hashtable: np.ndarray):
        d_hash_bucket = defaultdict(list)
        # d_hash_bucket = defaultdict(deque) # 经常append,deque比list快. 但declare时更慢
        buckets = [None] * n

        hashtable_bytes = hashtable.tobytes()
        itemsize = hashtable.dtype.itemsize
        rowsize = hashtable.shape[1] * itemsize

        for i in range(n):
            start_index = i * rowsize
            end_index = start_index + rowsize
            hashval = hashtable_bytes[start_index:end_index]
            tmp_bucket = d_hash_bucket[hashval]
            tmp_bucket.append(i)  # 这行要执行n次，太慢了
            buckets[i] = tmp_bucket
        return d_hash_bucket, buckets

    ls_dict, ls_buckets = zip(*map(map_func, hashtables))
    return ls_dict, ls_buckets  # TODO 以后跑更大一点的数据时，可以不用返回ls_dict


def count_idx_in_bucket(ls_buckets: list[list[set]]) -> list[Counter]:
    size = len(ls_buckets[0])  # size of the dataset

    ls_sample_buckets = zip(*ls_buckets)

    def map_func(i, sample_buckets):
        idxs_collision = itertools.chain.from_iterable(sample_buckets)
        tmp_counter = Counter(idxs_collision)
        del tmp_counter[i]

        return tmp_counter

    ls_d_idx_count = list(map(map_func, range(size), ls_sample_buckets))

    return ls_d_idx_count


def get_neighbor_idx_and_weight(
    X: np.ndarray, w: float = 1, t: int = 5, L: int = 10, seed: int = None
) -> tuple[list[list[int]], list[list[int]]]:
    """Get the neibor index and weight of each instance."""
    ls_buckets = convert_hashtable(lsh_2stable(X, w=w, t=t, L=L, seed=seed))[1]

    size = len(ls_buckets[0])  # size of the dataset
    ls_sample_buckets = zip(*ls_buckets)

    def map_func(i, sample_buckets):
        idxs_collision = itertools.chain.from_iterable(sample_buckets)
        tmp_counter = Counter(idxs_collision)
        del tmp_counter[i]
        nb_idxs, nb_weights = list(tmp_counter.keys()), list(tmp_counter.values())
        return nb_idxs, nb_weights

    ls_nb_idx, ls_nb_weight = zip(*map(map_func, range(size), ls_sample_buckets))

    return ls_nb_idx, ls_nb_weight


def calc_p_collision(dist: float, w: float = 1) -> float:
    """Calculate the probability of collision given the `w` of LSH and the `dist` between two points

    Args:
        dist (float): _description_
        w (float, optional): _description_. Defaults to 1.

    Returns:
        float: Probability.
    """
    if dist == 0:
        return 1
    elif dist < 0 or w <= 0:
        raise ValueError(
            f"`dist`, `w` should be all greater than 0. Now dist={dist}, w={w}"
        )

    c = dist
    v, err = integrate.quad(lambda t: np.e ** (-0.5 * (t / c) ** 2) * (1 - t / w), 0, w)
    # print(v,err)
    # TODO err is small for most cases now. But need to consider whether to limit the precision
    proba = np.sqrt(2 / (c * c * np.pi)) * v
    if proba < 0 or proba > 1:
        raise RuntimeError(
            f"Probaility is outside arange [0,1]. Now it is {proba}. dist={dist}, r={w}"
        )
    return proba


def calc_exp_collis(w: float, t: int, L: int, dist: float) -> float:
    """Calculate the expected number of collision of a point in a LSH with `w` and `t` and `L` hash tables."""
    return L * (calc_p_collision(dist, w) ** t)


def calc_ERk(k: int, n: int, d: int, R: float):
    r"""Calculate the expected radius the k-th nearest neighbor of a point in a d-dimensional space,
    when there are n points uniformly distributing in the space and the radius of the space is R.
    .. math:
        \mathbb{E}(R_k) = \frac{R\cdot k^{[1/l]}}{(N+1)^{1/l}}
    """
    return R * (special.poch(k, 1 / d) / special.poch(n + 1, 1 / d))


def get_lsh_ours_op_args(
    data: pd.DataFrame | np.ndarray,
    R_estim: float = None,
    use_op_arg_buffer: bool = False,
    k_of_dist1=30,
    k_of_dist2=60,
    seed=None,
):
    # X = preprocess_dataset(data)
    # dm = get_distance_matrix(X)
    # dm_sorted = sort_distance_matrix(dm)
    # dist1 = dm_sorted[:,0].mean()
    # dist2 = dm_sorted[:,29].mean()
    # num_instance = data.shape[0]
    dimension = data.shape[1]
    # R = op_args["R"]
    # R = dm_sorted[np.random.choice(dm_sorted.shape[0], 5, replace=False),29].mean()
    # dist1 = calc_ERk(1, 30, d, R)
    # dist2 = calc_ERk(30, 30, d, R)
    # C = 3
    # dist1 = R / C
    # dist2 = R
    R_estim = (
        estimate_dist_knn(X=data, k=k_of_dist1, seed=seed)
        if R_estim is None
        else R_estim
    )
    dist1 = R_estim  # R_30
    dist2 = R_estim * (
        special.poch(k_of_dist2, 1 / dimension)
        / special.poch(k_of_dist1, 1 / dimension)
    )  # R_60 = c R_30

    # N for small dataset. <10^4
    # math.sqrt(N) for most dataset. 10^4~10^8
    # math.log(N) for very large dataset. >10^8
    # if num_instance < 10000:
    #     coef = num_instance
    # elif num_instance < 100000000:
    #     coef = math.sqrt(num_instance)
    # else:
    #     coef = math.log(num_instance)
    coef = 1

    # if use buffer
    buffer_file_dir = "../tmp"
    buffer_file_path = os.path.join(buffer_file_dir, "LSH_op_arg_buffer.json")
    if use_op_arg_buffer:
        if os.path.exists(buffer_file_path):
            with open(buffer_file_path, "r", encoding="utf-8") as f:
                dict_hash_list = json.load(f)
        else:
            dict_hash_list = {}
        values = dict_hash_list.get(
            str(hash(dist1 + dist2 + coef)), None
        )  # 注意json的key只能是字符串
        if values is not None:
            logging.info("(%f,%f,%f) use the buffer %s", dist1, dist2, coef, values)
            return values[-3:]
        else:
            w, t, L, is_optimal = _get_lsh_ours_op_args(dist1, dist2, coef)
            if is_optimal:  # only write the result when the solver get the optimal
                os.makedirs(buffer_file_dir, exist_ok=True)
                with open(buffer_file_path, "w", encoding="utf-8") as f:
                    dict_hash_list[hash(dist1 + dist2 + coef)] = [
                        dist1,
                        dist2,
                        coef,
                        w,
                        t,
                        L,
                    ]
                    json.dump(
                        dict_hash_list, f
                    )  # 注意json的key只能是字符串，即使dict的key是int。保存时会默认转换
                logging.info(
                    "The buffer of lsh_ours_op_args is saved in %s", buffer_file_path
                )
    else:
        w, t, L, is_optimal = _get_lsh_ours_op_args(dist1, dist2, coef)

    return w, t, L


def _get_lsh_ours_op_args(dist1: float, dist2: float, coef: float = 1):
    dist1, dist2 = float(dist1), float(dist2)

    # check arg
    if dist1 > dist2:
        raise ValueError("dist1 should be smaller than dist2")

    def _obj_fun(x):
        w, t, L = x
        # return t*L
        return coef * calc_exp_collis(w, t, L, dist2)

    def _constraint_fun1(x):
        w, t, L = x
        return calc_exp_collis(w, t, L, dist1)

    # def _constraint_fun2(x):
    #     w, t, L = x
    #     return coef * calc_exp_collis(w, t, L, dist2)

    def _constraint_fun3(x):
        w, t, L = x
        return t * L

    w_bounds = (0.001, 10)
    t_bounds = (1, 100)
    L_bounds = (1, 100)
    tL_max = 100
    # TODO bound的速度会更快，只是经验上来说最优值会在里面。如果真的在外边则会有Warning

    # def _estimate_tL_from_w(dist1, dist2, num_instance, w):
    #     p1 = calc_p_collision(dist1, w)
    #     p2 = calc_p_collision(dist2, w)
    #     rho = math.log(p1,p2)
    #     t = math.log(num_instance, 1/p2)
    #     L = num_instance**(rho)/p1
    #     return t,L

    def _clip(val: float, lb: float, ub: float) -> float:
        return lb if val < lb else ub if val > ub else val

    w_init = dist2  # (dist1+dist2)/2
    w_init = _clip(w_init, *w_bounds)
    # t_init, L_init = _estimate_tL_from_w(dist1, dist2, num_instance, w_init)
    t_init, L_init = t_bounds[0], L_bounds[0]
    t_init, L_init = math.ceil(t_init), math.ceil(L_init)
    t_init = _clip(t_init, *t_bounds)
    L_init = _clip(L_init, *L_bounds)
    isOpt = True
    res = optimize.minimize(
        fun=_obj_fun,
        x0=[w_init, t_init, L_init],
        # args=(dist1, dist2),
        method="SLSQP",  # 有constraints时，默认是 SLSQP，其他都不行
        bounds=[w_bounds, t_bounds, L_bounds],
        constraints=[
            optimize.NonlinearConstraint(_constraint_fun1, 1, np.inf),
            # optimize.NonlinearConstraint(
            #     _constraint_fun2, 0, 1
            # ),  # 数值上的考虑，将coef放在不等式的左边
            optimize.NonlinearConstraint(_constraint_fun3, 1, tL_max),
        ],
        options={"disp": False},
    )

    w, t, L = res.x
    if not res.success:
        logging.warning(
            "(%f, %f, %f) fail to find the optimal in the first stage. (w, t, L)=%s."
            # 'Now reset them to their initial value (%f, %d, %d)',
            ,
            dist1,
            dist2,
            coef,
            res.x,
            # w_init, t_init, L_init
        )
        isOpt = False
        # w,t,L = w_init, t_init, L_init

    logging.info("(%f, %f, %f) first stage:w,t,L= %s", dist1, dist2, coef, (w, t, L))

    model = pyo.ConcreteModel()
    # model.w = pyo.Var(domain=pyo.PositiveReals, initialize=w)
    model.w = pyo.Param(initialize=w)

    # 虽然数学上多大都行，但还是限制个范围以防离谱的结果
    # model.t = pyo.Var(domain=pyo.PositiveIntegers, initialize=math.ceil(t))
    # model.L = pyo.Var(domain=pyo.PositiveIntegers, initialize=math.ceil(L))
    model.t = pyo.Var(
        domain=pyo.PositiveIntegers, initialize=math.ceil(t), bounds=t_bounds
    )
    model.L = pyo.Var(
        domain=pyo.PositiveIntegers, initialize=math.ceil(L), bounds=L_bounds
    )

    model.c1 = pyo.Constraint(expr=_constraint_fun1((model.w, model.t, model.L)) >= 1)
    # model.c2 = pyo.Constraint(
    #     expr=_constraint_fun2((model.w, model.t, model.L)) <= 1
    # )  # 数值上的考虑，将coef放在不等式的左边
    model.c3 = pyo.Constraint(
        expr=_constraint_fun3((model.w, model.t, model.L)) <= tL_max
    )

    model.objective = pyo.Objective(
        expr=_obj_fun((model.w, model.t, model.L)), sense=pyo.minimize
    )
    res = pyo.SolverFactory("mindtpy").solve(
        model, mip_solver="glpk", nlp_solver="ipopt"
    )
    w, t, L = pyo.value(model.w), pyo.value(model.t), pyo.value(model.L)
    if not (
        (res.solver.status == pyo.SolverStatus.ok)
        and (res.solver.termination_condition == pyo.TerminationCondition.optimal)
    ):
        logging.warning(
            "(%f, %f, %f) fail to find the optimal in the first stage. (w, t, L)=%s."
            # 'Now reset them to their initial value (%f, %d, %d)'
            ,
            dist1,
            dist2,
            coef,
            (pyo.value(model.w), pyo.value(model.t), pyo.value(model.L)),
            # w_init, t_init, L_init
        )
        # w, t, L = w_init, t_init, L_init
        isOpt = False

    logging.info("(%f, %f, %f) second stage:w,t,L= %s", dist1, dist2, coef, (w, t, L))

    logging.info(
        """
        dist1处期望碰撞次数: %f,
        dist2处期望碰撞次数: %f,
        小于dist1的点一次都没碰撞的概率 < %f,
        大于dist2的点至少碰撞一次的概率 < %f
        """,
        calc_exp_collis(w, t, L, dist1),
        calc_exp_collis(w, t, L, dist2),
        (1 - calc_p_collision(dist1, w) ** t) ** L,
        1 - (1 - calc_p_collision(dist2, w) ** t) ** L,
    )

    return w, int(t), int(L), isOpt


def estimate_dist_knn(X: np.ndarray, k=30, n_samples=30, seed=None) -> float:
    """Estimate the distance of k-th nearest neighbor of a point in a d-dimensional space."""

    if seed is None:
        # current implementation is not thread-safe when seed is None
        seed = np.random.get_state()[1][0]  # get the seed of the global random state
    rng = np.random.default_rng(seed)

    data_size = X.shape[0]
    k = min(k, data_size - 1)  # k should be less than the size of the dataset

    # n_samples should be less than the size of the dataset
    n_samples = min(n_samples, data_size)

    # random select `n_samples` samples
    idxs = rng.choice(X.shape[0], n_samples, replace=False)
    X_sample = X[idxs]

    # calcuate the distances between X_sample and X fast
    nbrs = NearestNeighbors().fit(X)
    # k+1 because the first one is itself
    distances, indexs = nbrs.kneighbors(
        X_sample, n_neighbors=k + 1, return_distance=True
    )

    dist_estim = distances[:, -1].mean()
    return dist_estim


def lsh_ours(
    X: np.ndarray,
    y: np.ndarray,
    w: float = 1,
    t: int = 5,
    L: int = 10,
    seed: int = None,
    use_weight=True,
    use_op_arg=True,
    use_op_arg_buffer=False,
    R_estim=None,
    **kwargs,
) -> np.ndarray:
    """PDOC-LSH: Our instance selection method based on LSH.

    if `use_op_arg` is True, the optimal arguments of LSH will be used and the `w`, `t`, `L` will be ignored.

    Args:
        X (np.ndarray): The dataset with shape (size, dimension)
        y (np.ndarray): The raw label of the dataset. Positive instance is 1 and negative instance is -1
        w (float, optional): parameter of pstable-LSH. Defaults to 1.
        t (int, optional): Times of lsh. g(x)=(h_1(x),...,h_t(x)). Defaults to 5.
        L (int, optional): Number of hash tables. Defaults to 10.
        seed (int, optional): Random seed to reproduce results.
        use_weight (bool, optional): Whether to use the count of collision as the weight of the instance. Defaults to True.
            In `/src/analysis_exp.analyze_ours_param`, True is better than False.
        use_op_arg (bool, optional): Whether to use optimal arguments of LSH. Defaults to True.
        use_op_arg_buffer (bool, optional): Whether to use buffer of optimal arguments of LSH. Defaults to False.
        R_estim (float, optional): The estimated distance of k-th nearest neighbor of a point in a d-dimensional space. Defaults to None.
        **kwargs: Other parameters for pdoc.pdoc.

    Returns:
        np.ndarray: The selected indexes.

    """

    # TODO if not use_weight, the count of collision may not be necessary to speed up.

    # check and convert the format of y
    y = utils.format_y_binary(y, neg_and_pos=True)

    if use_op_arg:
        w, t, L = get_lsh_ours_op_args(
            X,
            R_estim=R_estim,
            use_op_arg_buffer=use_op_arg_buffer,
            seed=seed,
        )
        w = float(w)
        t = int(t)
        L = int(L)

    hashtables = lsh_2stable(X, w=w, t=t, L=L, seed=seed)
    ls_dict, ls_buckets = convert_hashtable(hashtables)
    ls_d_idx_count = count_idx_in_bucket(ls_buckets)

    ser_lable_count = pd.Series(y).value_counts()
    default_proba_diff = (
        ser_lable_count.loc[1] - ser_lable_count.loc[-1]
    ) / ser_lable_count.sum()

    def _calc_proba_diff(d_idx_count: dict[int, int], y: np.ndarray) -> float:
        """Calculate the probability difference between the positive and negative instances.
        Based on the count of collision between the instance and its neighbors.

        Args:
            d_idx_count (dict): {index_that_collide: the_number_of_collision}
            y (np.ndarray): the raw label of the dataset. Positive instance is 1
                and negative instance is -1

        Returns:
            float: The probability difference between the positive and negative instances.
        """
        if len(d_idx_count) == 0:
            # 没有邻居，说明离点很远，可能是异常点; 也有可能是参数没选好
            # logging.warning(
            #     "The instance does not have a collision with any others. The instance may be a outlier."
            # )
            return default_proba_diff
        if use_weight:
            tmp_ls = [y[idx] * count for idx, count in d_idx_count.items()]
            all_colli_count = sum(d_idx_count.values())
            shrinkage_factor = all_colli_count / (all_colli_count + 1)
            res = (
                shrinkage_factor * (sum(tmp_ls) / all_colli_count)
                + (1 - shrinkage_factor) * default_proba_diff
            )
        else:
            tmp_ls = [y[idx] for idx, count in d_idx_count.items()]
            all_neibor_count = len(d_idx_count.values())
            shrinkage_factor = all_neibor_count / (all_neibor_count + 1)
            res = (
                shrinkage_factor * (sum(tmp_ls) / all_neibor_count)
                + (1 - shrinkage_factor) * default_proba_diff
            )
        return res

    proba_diff = np.array(
        [_calc_proba_diff(d_idx_count, y) for d_idx_count in ls_d_idx_count]
    )

    # check if some values in proba_diff are equal to default_proba_diff
    # if true, the instance may not have collision.
    if default_proba_diff in proba_diff:
        logging.warning("Some instances may not have a collision with any others.")

    # bewteen [-1, 1].
    # negative (positive) means the instance is on the wrong (right) side of the optimal decision boundary
    # positive means the instance is on the  side of the optimal decision boundary
    # it close to 0 means the instance is close to the optimal decision boundary
    pdocs = proba_diff * y
    idxs_selected = pdoc.pdoc(pdocs, seed=seed, **kwargs)

    return idxs_selected


def _test():
    """Test the runtime and correctness of the function."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    print(_get_lsh_ours_op_args(0.127851, 0.786885, 5000))

    X = np.random.rand(100, 2)
    y = np.random.randint(0, 2, 100)
    idxs_selected = lsh_ours(X, y, selection_rate=0.5)

    X_sub = X[idxs_selected]
    y_sub = y[idxs_selected]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(X[:, 0], X[:, 1], c=y)
    ax1.set(title="Original Datatset")
    ax2.scatter(X_sub[:, 0], X_sub[:, 1], c=y_sub)
    ax2.set(title="Selected Dataset")
    plt.suptitle(f"{X.shape}->{X_sub.shape}")
    plt.show()

    # test the runtime
    X = np.random.default_rng(0).normal(size=(20000, 200))
    _ = get_neighbor_idx_and_weight(X, w=1, t=5, L=100, seed=0)

    # r2 = c r1
    # the change of c
    ks = np.arange(1, 31)
    ds = np.arange(4, 20, 2)
    dfs = []
    for d in ds:
        df = pd.DataFrame(
            {
                "d": [d] * len(ks),
                "k": ks,
                "c": special.poch(ks + 30, 1 / d) / special.poch(ks, 1 / d),
            }
        )
        dfs.append(df)
    df = pd.concat(dfs, axis=0, ignore_index=True)
    sns.lineplot(data=df, x="k", y="c", hue="d")
    plt.show()


if __name__ == "__main__":
    _test()
