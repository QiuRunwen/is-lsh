from typing import Sequence
import warnings

import numpy as np
from scipy import stats
from sklearn import metrics

from sklearn.neighbors import NearestNeighbors

if __name__ == "__main__":
    import utils
else:
    try:
        from . import utils
    except ImportError:
        import utils


def estimate_proba(
    y: np.ndarray,
    neighbor_idxs: list[int] = None,
    neighbor_weights: np.ndarray = None,
    shrinkage_factor: float = None,
) -> tuple[float, float]:
    """Calculate the probability of positive and negative classes,
    given the labels of all data, the indexs of neighbors and their weights.

    Args:
        y (np.ndarray): data labels, 1 for positive class and -1 for negative class.
        neighbor_idxs (list[int], optional): Indexs of neighbors. Defaults to None.
        neighbor_weights (np.ndarray, optional): Weights of neighbors, which need to correspond to `neighbor_idxs`
            one by one. And unit of weight should be 1, since it adjusts probability estimation using naive Bayes.
            Defaults to None.

    Returns:
        tuple[float, float]: Probability of positive and negative classes.
    """
    assert np.isin([-1, 1], np.unique(y)).all()
    assert y.ndim == 1
    default_pos_proba = (y == 1).mean()
    default_neg_proba = 1 - default_pos_proba
    if neighbor_idxs is None or len(neighbor_idxs) == 0:
        return default_pos_proba, default_neg_proba
    if neighbor_weights is None:
        neighbor_weights = np.ones(len(neighbor_idxs))
        tmp_filt = y[neighbor_idxs] == 1
        tmp_pos_sum = np.where(tmp_filt, 1, 0).sum()
        tmp_neg_sum = np.where(~tmp_filt, 1, 0).sum()
    else:
        assert len(neighbor_weights) == len(neighbor_idxs)
        tmp_filt = y[neighbor_idxs] == 1
        tmp_pos_sum = np.where(tmp_filt, neighbor_weights, 0).sum()
        tmp_neg_sum = np.where(~tmp_filt, neighbor_weights, 0).sum()

    tmp_sum = tmp_pos_sum + tmp_neg_sum

    if shrinkage_factor is None:
        # neighbor_count = len(neighbor_idxs)
        # shrinkage_factor = neighbor_count / (neighbor_count + 1)
        shrinkage_factor = 1
    else:
        assert 0 <= shrinkage_factor <= 1
    pos_proba = tmp_pos_sum / tmp_sum * shrinkage_factor + default_pos_proba * (
        1 - shrinkage_factor
    )
    neg_proba = tmp_neg_sum / tmp_sum * shrinkage_factor + default_neg_proba * (
        1 - shrinkage_factor
    )
    return pos_proba, neg_proba


def estimate_arr_proba(
    y: np.ndarray,
    ls_neighbor_idxs: list[list[int]],
    ls_neighbor_weights: list[np.ndarray] = None,
    shrinkage_factors: list[float] = None,
) -> np.ndarray:
    """Calculate the probability of positive and negative classes,
    given the labels of all data, the indexs of neighbors and their weights.

    Args:
        y (np.ndarray): data labels, 1 for positive class and -1 for negative class.
        ls_neighbor_idxs (list[list[int]]): Each element is the indexs of neighbors of a sample.
        ls_neighbor_weights (list[np.ndarray], optional): Each element is the weights of neighbors of a sample.
            Defaults to None.

    Returns:
        np.ndarray: Probability of positive and negative classes. The shape is (len(ls_neighbor_idxs), 2).
    """
    if ls_neighbor_weights is None:
        ls_neighbor_weights = [None] * len(ls_neighbor_idxs)
    else:
        assert len(ls_neighbor_weights) == len(ls_neighbor_idxs)

    if shrinkage_factors is None:
        shrinkage_factors = [None] * len(ls_neighbor_idxs)
    else:
        assert len(shrinkage_factors) == len(ls_neighbor_idxs)

    def map_proba(neighbor_idxs, neighbor_weights, shrinkage_factor):
        return estimate_proba(
            y=y,
            neighbor_idxs=neighbor_idxs,
            neighbor_weights=neighbor_weights,
            shrinkage_factor=shrinkage_factor,
        )

    arr_proba = np.array(
        list(map(map_proba, ls_neighbor_idxs, ls_neighbor_weights, shrinkage_factors))
    )
    return arr_proba


def estimate_proba_diff(
    y: np.ndarray,
    ls_neighbor_idxs: list[list[int]],
    ls_neighbor_weights: list[np.ndarray] = None,
    shrinkage_factors: list[float] = None,
) -> np.ndarray:
    """Calculate the difference of probability of positive and negative classes,
    given the labels of all data, the indexs of neighbors and their weights.

    Args:
        y (np.ndarray): data labels, 1 for positive class and -1 for negative class.
        ls_neighbor_idxs (list[list[int]]): Each element is the indexs of neighbors of a sample.
        ls_neighbor_weights (list[np.ndarray], optional): Each element is the weights of neighbors of a sample.
            Defaults to None.

    Returns:
        np.ndarray: Difference of probability of positive and negative classes. The shape is (len(ls_neighbor_idxs),).
    """
    arr_proba = estimate_arr_proba(
        y=y,
        ls_neighbor_idxs=ls_neighbor_idxs,
        ls_neighbor_weights=ls_neighbor_weights,
        shrinkage_factors=shrinkage_factors,
    )
    arr_proba_diff = arr_proba[:, 0] - arr_proba[:, 1]
    return arr_proba_diff


def estimate_proba_diff_oppo_cls(
    y: np.ndarray,
    ls_neighbor_idxs: list[list[int]],
    ls_neighbor_weights: list[np.ndarray] = None,
    shrinkage_factors: list[float] = None,
    use_fast: bool = True,
    y_train: np.ndarray = None,
) -> np.ndarray:
    """Calculate the probability difference to opposite class,
    given the labels of all data, the indexs of neighbors and their weights.

    Args:
        y (np.ndarray): data labels, 1 for positive class and -1 for negative class.
        ls_neighbor_idxs (list[list[int]]): Each element is the indexs of neighbors of a sample.
        ls_neighbor_weights (list[np.ndarray], optional): Each element is the weights of neighbors of a sample.
            Defaults to None.

    Returns:
        np.ndarray: Probability difference to opposite class. The shape is (len(ls_neighbor_idxs),).
    """
    if use_fast:
        return _fast_estimate_proba_diff_oppo_cls(
            y=y,
            ls_neighbor_idxs=ls_neighbor_idxs,
            ls_neighbor_weights=ls_neighbor_weights,
            y_train=y_train,
        )
    else:
        if y_train is None:
            y_train = y

        arr_proba_diff = estimate_proba_diff(
            y=y_train,
            ls_neighbor_idxs=ls_neighbor_idxs,
            ls_neighbor_weights=ls_neighbor_weights,
            shrinkage_factors=shrinkage_factors,
        )
        # arr_proba_diff_oppo_cls = np.where(y == 1, arr_proba_diff, -arr_proba_diff)
        assert np.isin([-1, 1], np.unique(y)).all()
        arr_proba_diff_oppo_cls = y * arr_proba_diff
        return arr_proba_diff_oppo_cls


def _fast_estimate_proba_diff_oppo_cls(
    y: np.ndarray,
    ls_neighbor_idxs: list[list[int]],
    ls_neighbor_weights: list[np.ndarray] = None,
    y_train: np.ndarray = None,
):
    if y_train is None:
        y_train = y
    y_train = utils.format_y_binary(y_train, neg_and_pos=True)
    default_pos_proba = (y_train == 1).mean()
    default_neg_proba = 1 - default_pos_proba
    default_pd = default_pos_proba - default_neg_proba
    # if ls_neighbor_weights is None:
    #     ls_neighbor_weights = [
    #         np.ones(len(neighbor_idxs), dtype=np.int32)
    #         for neighbor_idxs in ls_neighbor_idxs
    #     ]

    # faster. but wrong for `np.sum(neighbor_weights)``
    # ls_neighbor_weights = [1] * len(ls_neighbor_idxs)
    # ls_neighbor_weights = np.ones(len(ls_neighbor_idxs), dtype=np.int32)

    if ls_neighbor_weights is None:

        def map_proba_diff_2(neighbor_idxs):
            if neighbor_idxs is None or len(neighbor_idxs) == 0:
                return default_pd
            else:
                return np.sum(y_train[neighbor_idxs]) / len(neighbor_idxs)

        arr_proba_diff = np.array(list(map(map_proba_diff_2, ls_neighbor_idxs)))
    else:

        def map_proba_diff(neighbor_idxs, neighbor_weights):
            if neighbor_idxs is None or len(neighbor_idxs) == 0:
                return default_pd
            else:
                if not isinstance(neighbor_weights, np.ndarray):
                    neighbor_weights = np.array(neighbor_weights)

                tmp_arr = np.where(
                    y_train[neighbor_idxs] == 1, neighbor_weights, -neighbor_weights
                )
                return tmp_arr.sum() / np.sum(neighbor_weights)

        arr_proba_diff = np.array(
            list(map(map_proba_diff, ls_neighbor_idxs, ls_neighbor_weights))
        )
    return y * arr_proba_diff


# def calc_proba_diff_oppo_cls(
#     X: np.ndarray,
#     y: np.ndarray,
#     method: str = "Gaussian",
#     **kwargs,
# ) -> np.ndarray:
#     """Calculate the probability difference to opposite class based on the distribution of each class.

#     Args:
#         X (np.ndarray): data matrix, shape is (n_samples, n_features).
#         y (np.ndarray): data labels, 1 for positive class and -1 for negative class.
#         method (str, optional): The method to calculate the probability difference to opposite class.
#             Defaults to "Gaussian". Supported methods: {"Gaussian"}.
#         **kwargs: The parameters of the method.

#     Returns:
#         np.ndarray: Probability difference to opposite class. The shape is (n_samples,).
#     """

#     if method == "Gaussian":
#         return calc_pdoc_gaussian(X=X, y=y, **kwargs)
#     else:
#         raise ValueError(f"method {method} is not supported.")


def calc_pdoc(
    y: np.ndarray,
    proba_x_mid_pos: np.ndarray,  # P(x|+1), likelihood
    proba_x_mid_neg: np.ndarray,  # P(x|-1), likelihood
    pos_proba_prior: float = 0.5,  # P(+1), prior
) -> np.ndarray:
    r"""
    pdoc(x,y) = y * ( P(+1|x) - P(-1|x) )
    = y * ( (P(x|-1)P(-1) - P(x|+1)P(+1)) / (P(x|-1)P(-1) + P(x|+1)P(+1)) )
    """
    assert (
        y.ndim == 1
        # Probabilities density is not necessary to be in [0, 1]
        # and ((proba_x_mid_pos <= 1) & (proba_x_mid_pos >= 0)).all()
        # and ((proba_x_mid_neg <= 1) & (proba_x_mid_neg >= 0)).all()
        # and 0 <= pos_proba_prior <= 1
    )
    neg_proba_prior = 1 - pos_proba_prior
    return (
        y
        * (proba_x_mid_pos * pos_proba_prior - proba_x_mid_neg * neg_proba_prior)
        / (proba_x_mid_pos * pos_proba_prior + proba_x_mid_neg * neg_proba_prior)
    )


def calc_pdoc_xor(
    X: np.ndarray,
    y: np.ndarray,
    sigma: float = 1,
    centers=[[1, 1], [-1, 1], [-1, -1], [1, -1]],
    pos_proba_prior: float = 0.5,
):
    assert X.ndim == 2 and y.ndim == 1 and len(centers) == 4
    means = np.array(centers)

    # calculate the probability density function (PDF) of the Gaussian distribution
    # P(x|+1) = 1- (1-P_1(x))* (1-P_3(x))
    # P(x|-1) = 1- (1-P_2(x))* (1-P_4(x))
    # P_1(x) = P(x|+1, center=(1,1))
    p1 = stats.multivariate_normal.pdf(X, mean=means[0], cov=sigma)
    p2 = stats.multivariate_normal.pdf(X, mean=means[1], cov=sigma)
    p3 = stats.multivariate_normal.pdf(X, mean=means[2], cov=sigma)
    p4 = stats.multivariate_normal.pdf(X, mean=means[3], cov=sigma)

    proba_x_mid_pos = 1 - (1 - p1) * (1 - p3)
    proba_x_mid_neg = 1 - (1 - p2) * (1 - p4)

    return calc_pdoc(
        y=y,
        proba_x_mid_pos=proba_x_mid_pos,
        proba_x_mid_neg=proba_x_mid_neg,
        pos_proba_prior=pos_proba_prior,
    )


def calc_pdoc_gaussian(
    X: np.ndarray,
    y: np.ndarray,
    neg_mean: float | Sequence[float] = None,
    neg_cov: float | Sequence[Sequence[float]] = None,
    pos_mean: float | Sequence[float] = None,
    pos_cov: float | Sequence[Sequence[float]] = None,
    pos_proba_prior: float = None,
):
    """Calculate the probability difference to opposite class based on Gaussian distribution.

    Args:
        X (np.ndarray): data matrix, shape is (n_samples, n_features).
        y (np.ndarray): data labels, 1 for positive class and -1 for negative class.
        neg_mean (float | Sequence[float]): The mean of negative class.
        neg_cov (float | Sequence[Sequence[float]]): The covariance of negative class.
        pos_mean (float | Sequence[float]): The mean of positive class.
        pos_cov (float | Sequence[Sequence[float]]): The covariance of positive class.
        pos_proba_prior (float): The prior probability of positive class.

    Returns:
        np.ndarray: Probability difference to opposite class. The shape is (n_samples,).

    """

    # check args
    assert X.ndim == 2 and y.ndim == 1
    y = utils.format_y_binary(y, neg_and_pos=True)
    neg_mean = X[y == -1].mean(axis=0) if neg_mean is None else neg_mean
    neg_cov = np.cov(X[y == -1].T) if neg_cov is None else neg_cov
    pos_mean = X[y == 1].mean(axis=0) if pos_mean is None else pos_mean
    pos_cov = np.cov(X[y == 1].T) if pos_cov is None else pos_cov
    pos_proba_prior = (
        (y == 1).sum() / y.size if pos_proba_prior is None else pos_proba_prior
    )

    # supposed multivariate normal distribution
    # calcuate probability of the area

    # ----- The count version -----
    # num_neg = (y == -1).sum()
    # num_pos = (y == 1).sum()

    # calcuate probability of the area according to probability density function and the number of samples
    # e.g. probability density value, pdf_neg(a) = 0.2, pdf_pos(a) = 0.5
    # the expected number of negative samples N_neg in the area `a` is `0.2 * num_neg`
    # the expected number of positive samples N_pos in the area `a` is `0.5 * num_pos`
    # the negative probability of the area `a` is `N_neg/(N_neg+N_pos)`
    # exp_neg_count_per_area = (
    #     stats.multivariate_normal.pdf(X, mean=neg_mean, cov=neg_cov).flatten() * num_neg
    # )
    # exp_pos_count_per_area = (
    #     stats.multivariate_normal.pdf(X, mean=pos_mean, cov=pos_cov).flatten() * num_pos
    # )
    # exp_count_per_area = exp_neg_count_per_area + exp_pos_count_per_area

    # neg_proba = exp_neg_count_per_area / exp_count_per_area
    # pos_proba = exp_pos_count_per_area / exp_count_per_area

    # ----- The probability version -----
    # The same as the count version, but it's faster and more accurate.
    # `P(+1|x) - P(-1|x) = (P(x|-1)P(-1) - P(x|+1)P(+1)) / (P(x|-1)P(-1) + P(x|+1)P(+1))`
    # based on the Bayes' theorem and Total probability theorem.
    pos_proba_prior = (
        (y == 1).sum() / y.size if pos_proba_prior is None else pos_proba_prior
    )
    neg_proba_prior = 1 - pos_proba_prior

    pos_tmp = (
        stats.multivariate_normal.pdf(X, mean=pos_mean, cov=pos_cov) * pos_proba_prior
    )
    neg_tmp = (
        stats.multivariate_normal.pdf(X, mean=neg_mean, cov=neg_cov) * neg_proba_prior
    )
    proba_diff = (pos_tmp - neg_tmp) / (neg_tmp + pos_tmp)

    return proba_diff * y


def find_optimal_decision_boundary(
    X: np.ndarray,
    y: np.ndarray,
    neg_mean: float | Sequence[float],
    neg_cov: float | Sequence[Sequence[float]],
    pos_mean: float | Sequence[float],
    pos_cov: float | Sequence[Sequence[float]],
    num_grid: int = 100,
    epsilon: float = 0.0001,
) -> tuple[float, float]:
    """Find the optimal decision boundary based on the distribution of each class.
    find the `x` given `num_neg*pdf1(x) = num_pos*pdf2(x)`
    """

    # Numerical solution
    x_min = X.min(axis=0)  # [x1_min, x2_min, ...]
    x_max = X.max(axis=0)  # [x1_max, x2_max, ...]

    x_points_diagonal = np.linspace(
        x_min, x_max, num_grid
    )  # When X is 2d, x_grid are the diagonal points of the grid
    x_high_dimension_grid = np.meshgrid(*x_points_diagonal.T)
    x_points = np.vstack([col_grid.ravel() for col_grid in x_high_dimension_grid]).T

    y_pos_num = (y == 1).sum()
    y_neg_num = len(y) - y_pos_num
    # coefficent_neg = y_neg_num / (y_neg_num + y_pos_num)

    # def func(x):
    #     return coefficent_neg * stats.multivariate_normal.pdf(
    #         x, mean=neg_mean, cov=neg_cov
    #     ) - stats.multivariate_normal.pdf(x, mean=pos_mean, cov=pos_cov)

    # [multivariate] https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html
    # [multivariate] https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html
    # 只能找一个解，没用
    # guess = np.zeros(X.shape[1], dtype=np.float64)
    # sol = optimize.root(func, x0=guess, method="hybr")
    # res = sol.x

    y_grid = np.ones(x_points.shape[0])

    y_neg_gird_num = int(len(y_grid) * (y_neg_num / len(y)))
    y_grid[-y_neg_gird_num:] = -1

    pdocs = calc_pdoc_gaussian(
        X=x_points,
        y=y_grid,
        neg_mean=neg_mean,
        neg_cov=neg_cov,
        pos_mean=pos_mean,
        pos_cov=pos_cov,
        pos_proba_prior=y_pos_num / len(y),
    )

    # formular solution
    # New random variable = X_pos - X_neg
    # parameters of new random variable = N_pos* mean_pos - N_neg*mean_neg, N_pos^2 * cov_pos + N_neg^2 * cov_neg

    # return x_points[np.abs(proba_diff) < epsilon]
    return x_points[np.abs(pdocs) < epsilon]


def score_estimate(
    pdoc_true: np.ndarray,
    pdoc_estim: np.ndarray,
    pdoc_true_sgn: np.ndarray = None,
    pdoc_estim_sgn: np.ndarray = None,
    metric: str | list = "Precision",
) -> float:
    """Calculate the score of the estimated pdoc.

    Args:
        pdoc_true (np.ndarray): The true pdoc.
        pdoc_estim (np.ndarray): The estimated pdoc.
        pdoc_true_sgn (np.ndarray, optional): sgn(pdoc_true). if None, will be calculated. Defaults to None.
        pdoc_estim_sgn (np.ndarray, optional): sgn(pdoc_estim). if None, will be calculated. Defaults to None.
        metric (str | list, optional): The metric to calculate the score. Defaults to "Precision". Supported metrics:
            "F1", "AUC", "Accuracy", "Precision", "Recall", "MSE", "MAE", "RMSE", "MAPE".

    Returns:
        float | list[float]: The score of the estimated pdoc.
    """

    def sgn(arr: np.ndarray):
        """Negative: -1, Zero: 1, Positive: 1"""
        return np.sign(arr) + (arr == 0)

    if pdoc_true_sgn is None:
        pdoc_true_sgn = sgn(pdoc_true)
    if pdoc_estim_sgn is None:
        pdoc_estim_sgn = sgn(pdoc_estim)

    if isinstance(metric, str):
        metric = [metric]
    elif not isinstance(metric, list):
        raise ValueError(f"metric {metric} is not supported.")

    def _score(metric):
        # default argument pos_label=1
        if metric == "F1":
            return metrics.f1_score(y_true=pdoc_true_sgn, y_pred=pdoc_estim_sgn)
        elif metric == "AUC":
            only_one_val = np.unique(pdoc_true_sgn).size == 1
            if only_one_val:
                warnings.warn(
                    "The number of unique values of pdoc_true is 1, AUC will be None."
                )
                return None
            return metrics.roc_auc_score(y_true=pdoc_true_sgn, y_score=pdoc_estim)
        elif metric == "Accuracy":
            return metrics.accuracy_score(y_true=pdoc_true_sgn, y_pred=pdoc_estim_sgn)
        elif metric == "Precision":
            return metrics.precision_score(y_true=pdoc_true_sgn, y_pred=pdoc_estim_sgn)
        elif metric == "Recall":
            return metrics.recall_score(y_true=pdoc_true_sgn, y_pred=pdoc_estim_sgn)
        elif metric == "MSE":
            return metrics.mean_squared_error(y_true=pdoc_true, y_pred=pdoc_estim)
        elif metric == "MAE":
            return metrics.mean_absolute_error(y_true=pdoc_true, y_pred=pdoc_estim)
        elif metric == "RMSE":
            return metrics.mean_squared_error(pdoc_true, pdoc_estim, squared=False)
        elif metric == "MAPE":
            return metrics.mean_absolute_percentage_error(pdoc_true, pdoc_estim)
        else:
            raise ValueError(f"metric {metric} is not supported.")

    res = [_score(m) for m in metric]
    if len(res) == 1:
        return res[0]
    else:
        return res


def eval_estimate(
    pdoc_true: np.ndarray, pdoc_estim: np.ndarray, metric=None
) -> dict[str, float]:
    """Evaluate the estimated pdoc. If metric is None, will calculate all metrics.

    return:
        dict[str, float]: The score of the estimated pdoc.
    """

    if metric is None:
        metric = [
            "F1",
            "AUC",
            "Accuracy",
            "Precision",
            "Recall",
            "MSE",
            "MAE",
            "RMSE",
            "MAPE",
        ]
    scores = score_estimate(pdoc_true, pdoc_estim, metric=metric)
    res = dict(zip(metric, scores))
    return res


def get_ls_neighbor_idxs(
    X: np.ndarray,
    n_neighbors: int = None,
    radius: float = None,
    p_norm: int = 2,
    return_distance: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Get the indexs of neighbors of each sample, using the distance matrix of p_norm.

    Args:
        X (np.ndarray): data matrix, shape is (n_samples, n_features).
        n_neighbors (int, optional): the k nearest neighbors. Defaults to None, which means use radius.
        radius (float, optional): The minimum distance that can be considered as a neighbor. Defaults to 0.5.
        p_norm (int, optional): The p-norm of distance, where 2 is the Euclidean distance. Defaults to 2.
        return_distance (bool, optional): Whether to return the distances. Defaults to False.

    Returns:
        np.ndarray[np.ndarray] | tuple[np.ndarray, np.ndarray]: The indexs of neighbors of each sample and the distances.
    """

    model_nn = NearestNeighbors(
        algorithm="brute", metric="minkowski", p=p_norm, n_jobs=-1
    )
    model_nn.fit(X)

    if n_neighbors is not None:
        if radius is not None:
            warnings.warn(
                "n_neighbors is not None and radius is not None, radius will be ignored."
            )
        res = model_nn.kneighbors(
            n_neighbors=n_neighbors, return_distance=return_distance
        )
    elif radius is not None:
        res = model_nn.radius_neighbors(radius=radius, return_distance=return_distance)
    else:
        raise ValueError("n_neighbors and radius cannot be both None.")

    if return_distance:
        return res[1], res[0]  # indexs, distances
    else:
        return res


def get_neighbor_idx_and_weight(
    X, n_neighbors: int = None, radius: float = None, p_norm: int = 2
) -> tuple[list[list[int]], list[list[float]]]:
    """Get the neibor index and weight of each instance."""
    arr_neibor_idx, arr_neibor_dist = get_ls_neighbor_idxs(
        X=X, n_neighbors=n_neighbors, radius=radius, p_norm=p_norm, return_distance=True
    )

    # 1/distance as weight, if distance is 0, set weight to the maximum in the array
    for arr_dist in arr_neibor_dist:
        if arr_dist.size != 0:
            tmp_filt = arr_dist == 0
            if tmp_filt.all():
                arr_dist[:] = 1
            else:
                arr_dist[arr_dist == 0] = arr_dist[arr_dist != 0].max()
                arr_dist[:] = 1 / arr_dist

    return arr_neibor_idx, arr_neibor_dist


def sample_by_weight(
    data: np.ndarray,
    weights: np.ndarray,
    size: int,
    oversampling: bool = False,
    method: str = "filter",
    seed: int = None,
) -> np.ndarray:
    """Sample data by weights.

    Args:
        data (np.ndarray): _description_
        weights (np.ndarray): The weights of samples. The negative weights will be set to 0.
        size (int): The number of samples to select.
        oversampling (bool, optional): if use oversampling when the available number of samples (weights!=0) is not enough.
            If False, the number of samples will be less than `size`. If True, the number of samples will be equal to `size`.
            Defaults to False.
        method (str, optional): The method to select samples {"random", "filter"}.
            Defaults to "filter".

    Raises:
        ValueError: _description_

    Returns:
        np.ndarray: The selected samples.
    """

    assert data.shape[0] == weights.shape[0] and weights.ndim == 1
    if size > data.shape[0]:
        warnings.warn("size is larger than the number of samples, will return all.")
        return data

    weights = weights.copy()
    tmp_filt = weights < 0
    if tmp_filt.any():
        warnings.warn("weights contains negative values, will be set to 0.")
        weights[tmp_filt] = 0

    weight_sum = weights.sum()
    if weight_sum == 0:
        warnings.warn("The sum of weights is 0. Will return empty.")
        return np.array([])

    if seed is None:
        # current implementation is not thread-safe when seed is None
        seed = np.random.get_state()[1][0]  # get the seed of the global random state
    rng = np.random.default_rng(seed)

    weights = weights / weight_sum

    avail_num = np.sum(weights > 0)
    num_to_supplement = max(size - avail_num, 0)

    all_idxs = np.arange(data.shape[0])
    if num_to_supplement == 0:
        # the number of available samples is enough
        if method == "random":
            # Weighted random sampling
            sample_idxs = rng.choice(
                all_idxs, size=size, replace=False, p=weights, shuffle=False
            )
        elif method == "filter":
            # choose the samples with the largest weights
            sample_idxs = all_idxs[np.argsort(weights)[-size:]]
        else:
            raise ValueError(f"method {method} is not supported.")
    else:
        # the number of available samples is not enough
        # Oversampling the samples with weights > 0
        # TODO oversampling may cause overfitting
        if oversampling:
            if method == "random":
                # Weighted random sampling
                sample_idxs = rng.choice(
                    all_idxs, size=size, replace=True, p=weights, shuffle=False
                )
            elif method == "filter":
                # choose the samples with the largest weights
                sample_idxs = all_idxs[np.argsort(weights)[-size:]]
            else:
                raise ValueError(f"method {method} is not supported.")
        else:
            # Use all samples whose weight>0.
            sample_idxs = all_idxs[weights > 0]

            # supplement_idxs = np.random.choice(
            #     all_idxs[weights == 0], size=num_to_supplement, replace=False
            # )
            # sample_idxs = np.hstack([sample_idxs, supplement_idxs])

    return data[sample_idxs]


def pdoc2weight(arr_pdoc: np.ndarray, close1better=True, sumas1=False) -> np.ndarray:
    """Convert pdoc to weight."""
    assert arr_pdoc.ndim == 1
    assert -1 <= arr_pdoc.min() and arr_pdoc.max() <= 1

    # min-max normalization
    if close1better:
        # pdoc_min = -1
        # pdoc_max = 1
        # weights = (arr_pdoc - pdoc_min) / (pdoc_max - pdoc_min)
        weights = (arr_pdoc + 1) / 2
    else:
        arr_pdoc = np.abs(arr_pdoc)
        # pdoc_min = 0
        # pdoc_max = 1
        # weights = 1 - (arr_pdoc - pdoc_min) / (pdoc_max - pdoc_min)
        # weights = (pdoc_max - arr_pdoc) / (pdoc_max - pdoc_min)
        weights = 1 - arr_pdoc

    if sumas1:
        weights = weights / weights.sum()

    return weights


def pdoc(
    pdocs: np.ndarray,
    thresholds: tuple[float, float] | list[tuple[float, float]] = (0, 1),
    selection_rate: float = 0.1,
    selection_type: str = "near_random",
    seed: int = None,
) -> np.ndarray:
    """Instance selection method based on the probability difference to opposite class (PDOC).
    -1 <= pdoc <= 1, pdoc close to 0 means the sample is near the decision boundary and hard to classify.

    1. if `thresholds` is not None, the samples will be selected by the thresholds. else not.
    2. the selected sample further be selected to reach `selection_rate` if `selection_rate` is not None.
        and the available samples are enough.
    3. `selection_rate` is not None. Then `selection_type` is used as the method to select samples.
        - "near_filter": select samples near the decision boundary by filter.
        - "near_random": select samples near the decision boundary by weighted random sampling.
        - "far_filter": select samples far from the decision boundary by filter.
        - "far_random": select samples far from the decision boundary by weighted random sampling.
        - "all_random": select samples by equal random sampling.

    In `/src/analysis_exp.analyze_pdoc_param`. Most cases, performance: `near` >= `far`, `random` >= `filter`.

    Args:
        pdocs (np.ndarray): The probability difference to opposite class (PDOC) of each sample.
        thresholds (tuple[float, float] | list[tuple[float, float]], optional): Select samples with pdoc in the range.
            e.g. thresholds = (-0.1, 0.1) means select samples with pdoc in [-0.1, 0.1].
                thresholds = [(-0.1, 0.1), (-0.2, 0.2)] means select samples with pdoc in [-0.1, 0.1] or [-0.2, 0.2].
            Defaults to None.
        selection_rate (float, optional): The selection rate of samples. Defaults to None.
        selection_type (str, optional): The method to select samples {"near_filter", "near_random", "far_filter", "far_random", "all_random"}.
        seed (int, optional): _description_. Defaults to None.

    Returns:
        np.ndarray: The indexs of selected samples.
    """

    # check pdocs
    assert pdocs.ndim == 1
    num_of_samples = pdocs.size

    if thresholds is None and selection_rate is None:
        warnings.warn("thresholds and selection_rate are both None, will select all.")
        return np.arange(num_of_samples)

    # use the threshold or selection rate to select samples
    # -1 <= pdoc <= 1
    # pdoc close to 0 means the sample is hard to classify,
    # also means the sample is close to the decision boundary

    bool_filter = np.ones(num_of_samples, dtype=bool)  # select all samples
    if thresholds is not None:
        if not isinstance(thresholds[0], (tuple, list)):
            thresholds = [thresholds]
        for threshold in thresholds:
            pdoc_min, pdoc_max = threshold
            assert -1 <= pdoc_min <= pdoc_max <= 1
            # bool_filter = bool_filter & ((pdocs >= pdoc_min) & (pdocs <= pdoc_max))
            bool_filter = (pdocs >= pdoc_min) & (pdocs <= pdoc_max)

    if selection_rate is None:
        selected_idxs = np.where(bool_filter)[0]
    else:
        # the number of samples to select
        selection_num = np.ceil(selection_rate * num_of_samples).astype(int)
        all_idxs = np.arange(num_of_samples)

        # TODO many false in bool_filter, performance can be improved
        prefer_loc, sample_method = selection_type.split("_")
        if prefer_loc in ("near", "far"):
            weights = np.where(
                bool_filter,
                pdoc2weight(pdocs, close1better=(prefer_loc == "far"), sumas1=False),
                0,
            )
        elif prefer_loc == "all":
            weights = np.where(bool_filter, 1, 0)
        else:
            raise ValueError(f"prefer_loc {prefer_loc} is not supported.")

        selected_idxs = sample_by_weight(
            all_idxs,
            weights=weights,
            size=selection_num,
            oversampling=False,
            method=sample_method,
            seed=seed,
        )

    return selected_idxs


def pdoc_knn(
    X: np.ndarray,
    y: np.ndarray,
    n_neighbors: int = 30,
    **kwargs,
):
    """Instance selection method based on the probability difference to opposite class (PDOC).
    PDOC is calculated based on k nearest neighbors.

    Args:
        X (np.ndarray): data matrix, shape is (n_samples, n_features).
        y (np.ndarray): data labels, 1 for positive class and -1 for negative class.
        n_neighbors (int, optional): The number of neighbors. Defaults to 30.
        **kwargs: The parameters of the method to select samples :func: `pdoc`.

    Returns:
        np.ndarray: The indexs of selected samples.
    """
    ls_neighbor_idxs = get_ls_neighbor_idxs(
        X, n_neighbors=n_neighbors, radius=None, p_norm=2, return_distance=False
    )
    pdocs = estimate_proba_diff_oppo_cls(
        y,
        ls_neighbor_idxs,
        ls_neighbor_weights=None,
        shrinkage_factors=None,
        use_fast=True,
        y_train=None,
    )

    selected_idxs = pdoc(pdocs, **kwargs)

    return selected_idxs


def pdoc_rnn(
    X: np.ndarray,
    y: np.ndarray,
    radius: float = 1,
    **kwargs,
):
    """Instance selection method based on the probability difference to opposite class (PDOC).
    PDOC is calculated based on radius nearest neighbors.

    Args:
        X (np.ndarray): data matrix, shape is (n_samples, n_features).
        y (np.ndarray): data labels, 1 for positive class and -1 for negative class.
        radius (float, optional): The radius of neighbors. Defaults to 1.
        **kwargs: The parameters of the method to select samples :func: `pdoc`.

    Returns:
        np.ndarray: The indexs of selected samples.
    """
    ls_neighbor_idxs = get_ls_neighbor_idxs(
        X, n_neighbors=None, radius=radius, p_norm=2, return_distance=False
    )

    pdocs = estimate_proba_diff_oppo_cls(
        y,
        ls_neighbor_idxs,
        ls_neighbor_weights=None,
        shrinkage_factors=None,
        use_fast=True,
    )

    selected_idxs = pdoc(pdocs, **kwargs)

    return selected_idxs


def pdoc_gb(
    X: np.ndarray,
    y: np.ndarray,
    neg_mean: float | Sequence[float] = None,
    neg_cov: float | Sequence[Sequence[float]] = None,
    pos_mean: float | Sequence[float] = None,
    pos_cov: float | Sequence[Sequence[float]] = None,
    pos_proba_prior: float = None,
    **kwargs,
) -> np.ndarray:
    """Instance selection method based on the probability difference to opposite class (PDOC).
    PDOC is calculated based on Gaussian distribution and Bayes' theorem.

    :func: `calc_pdoc_gaussian` is used to calculate pdoc using `neg_mean`, `neg_cov`, `pos_mean`,
        `pos_cov`, `pos_proba_prior`.
    :func: `pdoc` is used to select samples using `kwargs`.
    """
    pdocs = calc_pdoc_gaussian(
        X=X,
        y=y,
        neg_mean=neg_mean,
        neg_cov=neg_cov,
        pos_mean=pos_mean,
        pos_cov=pos_cov,
        pos_proba_prior=pos_proba_prior,
    )

    selected_idxs = pdoc(pdocs, **kwargs)

    return selected_idxs
