"""
Synthetic data generation functions.
"""

import numbers
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.utils import check_random_state, shuffle as util_shuffle
from scipy import stats


def gen_moons(*, n_samples: int = 5000, sigma: float = 0.2, seed: int = None):
    """Make two interleaving half circles.
    Modified from `sklearn.datasets.make_moons`.
    Make the radius of the two moons equal to 4 , and the center are at (-2, -1) and (2, 1).
    so the distance between optimal decision boundary and each point is about 1.
    It matches the `gen_blobs` dataset.
    """

    shuffle = True
    if isinstance(n_samples, numbers.Integral):
        n_samples_out = n_samples // 2
        n_samples_in = n_samples - n_samples_out
    else:
        try:
            n_samples_out, n_samples_in = n_samples
        except ValueError as e:
            raise ValueError(
                "`n_samples` can be either an int or a two-element tuple."
            ) from e

    generator = check_random_state(seed)

    outer_circ_x = 4 * np.cos(np.linspace(0, np.pi, n_samples_out)) - 2
    outer_circ_y = 4 * np.sin(np.linspace(0, np.pi, n_samples_out)) - 1
    inner_circ_x = 4 * np.cos(np.linspace(np.pi, 2 * np.pi, n_samples_in)) + 2
    inner_circ_y = 4 * np.sin(np.linspace(np.pi, 2 * np.pi, n_samples_in)) + 1

    X = np.vstack(
        [np.append(outer_circ_x, inner_circ_x), np.append(outer_circ_y, inner_circ_y)]
    ).T
    y = np.hstack(
        [np.zeros(n_samples_out, dtype=np.intp), np.ones(n_samples_in, dtype=np.intp)]
    )

    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

    if sigma is not None:
        X += generator.normal(scale=sigma, size=X.shape)
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["y"] = y
    return df, "y"


def gen_circles(*, n_samples: int = 5000, sigma: float = 0.2, seed: int = None):
    """Make a large circle containing a smaller circle in 2d.
    Modified from `sklearn.datasets.make_circles`.
    Make outer and inner circles with radius 3 and 1.
    Therefore the distance between optimal decision boundary and each point is 1,
    matching the `gen_blobs` dataset.
    """

    shuffle = True
    if isinstance(n_samples, numbers.Integral):
        n_samples_out = n_samples // 2
        n_samples_in = n_samples - n_samples_out
    else:  # n_samples is a tuple
        if len(n_samples) != 2:
            raise ValueError("When a tuple, n_samples must have exactly two elements.")
        n_samples_out, n_samples_in = n_samples

    generator = check_random_state(seed)
    # so as not to have the first point = last point, we set endpoint=False
    linspace_out = np.linspace(0, 2 * np.pi, n_samples_out, endpoint=False)
    linspace_in = np.linspace(0, 2 * np.pi, n_samples_in, endpoint=False)
    outer_circ_x = 3 * np.cos(linspace_out)
    outer_circ_y = 3 * np.sin(linspace_out)
    inner_circ_x = np.cos(linspace_in)
    inner_circ_y = np.sin(linspace_in)

    X = np.vstack(
        [np.append(outer_circ_x, inner_circ_x), np.append(outer_circ_y, inner_circ_y)]
    ).T
    y = np.hstack(
        [np.zeros(n_samples_out, dtype=np.intp), np.ones(n_samples_in, dtype=np.intp)]
    )
    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

    if sigma is not None:
        X += generator.normal(scale=sigma, size=X.shape)

    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["y"] = y
    return df, "y"


def gen_blobs(
    *,
    n_samples: int | tuple[int, int] = 5000,
    sigma: float | Sequence[float] = 0.2,
    dimension: int = 2,
    seed: int = None,
):
    """Generate isotropic Gaussian blobs for clustering."""
    if isinstance(n_samples, float):
        n_samples = int(n_samples)
    if isinstance(dimension, float):
        dimension = int(dimension)

    centers = [
        [-1] + [0] * (dimension - 1),
        [1] + [0] * (dimension - 1),
    ]
    # with return_centers = False, always returns a 2-tuple
    # pylint: disable-next=unbalanced-tuple-unpacking
    X, y = make_blobs(
        n_samples=n_samples,
        centers=centers,
        n_features=2,
        cluster_std=sigma,
        random_state=seed,
        return_centers=False,
    )
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(1, dimension + 1)])
    df["y"] = y
    return df, "y"


def gen_normal_precise_1d(*, n_samples: int = 5000, sigma: float = 1, seed: int = None):
    """Generate 1-d isotropic Gaussian blobs for clustering.
    Each point will fall in the center of a grid"""

    loc_neg = -1
    loc_pos = 1
    half_interval = 3 * sigma
    x_grid_neg = np.linspace(
        loc_neg - half_interval, loc_neg + half_interval, n_samples // 10
    )
    x_grid_pos = np.linspace(
        loc_pos - half_interval, loc_pos + half_interval, n_samples // 10
    )

    # Given the grid, we can compute the probability density function (PDF) of the Gaussian distribution.
    if sigma > 0:
        proba_density_neg = stats.norm.pdf(x_grid_neg, loc=loc_neg, scale=sigma)
        proba_density_pos = stats.norm.pdf(x_grid_pos, loc=loc_pos, scale=sigma)
    elif sigma == 0:
        proba_density_neg = np.ones_like(x_grid_neg)
        proba_density_pos = np.ones_like(x_grid_pos)
    else:
        raise ValueError("Sigma should be non-negative")

    n_samples_neg = n_samples // 2
    n_samples_pos = n_samples - n_samples_neg

    neg_nums = (
        (proba_density_neg * n_samples_neg / np.sum(proba_density_neg))
        .round()
        .astype(int)
    )
    pos_nums = (
        (proba_density_pos * n_samples_pos / np.sum(proba_density_pos))
        .round()
        .astype(int)
    )

    neg_samples = np.repeat(x_grid_neg, neg_nums)
    pos_samples = np.repeat(x_grid_pos, pos_nums)

    df = pd.DataFrame(
        {
            "x1": np.concatenate([neg_samples, pos_samples]),
            "y": np.concatenate(
                [
                    np.zeros_like(neg_samples, dtype=int),
                    np.ones_like(pos_samples, dtype=int),
                ]
            ),
        }
    )
    return df, "y"


def gen_xor(*, n_samples: int = 5000, sigma: float = 0.2, seed: int = None):
    """Generate 4 isotropic Gaussian blobs for clustering in 2d which form an XOR pattern."""
    # with return_centers = False, always returns a 2-tuple
    # pylint: disable-next=unbalanced-tuple-unpacking
    X, y = make_blobs(
        n_samples=n_samples,
        centers=[[1, 1], [-1, -1], [1, -1], [-1, 1]],
        n_features=2,
        cluster_std=sigma,
        random_state=seed,
        return_centers=False,
    )
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["y"] = y
    df["y"] = df["y"].map({0: 1, 1: 1, 2: 0, 3: 0})

    return df, "y"


def _test():
    """Test synthetic data generation functions."""
    import seaborn as sns
    import matplotlib.pyplot as plt

    gen_funs = (
        gen_circles,
        gen_blobs,
        gen_normal_precise_1d,
        gen_xor,
        gen_moons,
    )
    num_gen_funs = len(gen_funs)
    n_cols = 3
    n_rows = int(np.ceil(num_gen_funs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten()
    for gen_fun, ax in zip(gen_funs, axes):
        df, y_col = gen_fun(n_samples=5000, seed=42)
        print(gen_fun.__name__, df.shape)
        if df.shape[1] == 2:
            ax = sns.histplot(df, x="x1", hue=y_col, ax=ax)
        elif df.shape[1] == 3:
            ax = sns.scatterplot(data=df, x="x1", y="x2", hue=y_col, ax=ax)
        else:
            pass
        ax.set_title(gen_fun.__name__)
    plt.show()


if __name__ == "__main__":
    _test()
