"""
Analysis of groud true PDOC and estimated PDOC in dataset. 
Analysis of the experiment results that produced by `exp_pdoc.py`.
"""

from typing import Sequence
import itertools
import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from matplotlib.patches import Circle
import joblib

from datasets import load_one
from ismethod import pdoc
from ismethod.utils import format_y_binary
import estim_pdoc

# sns.set_theme(style="whitegrid")
sns.reset_defaults()


def read_and_preprocess(result_dir: str):
    df = pd.read_csv(os.path.join(result_dir, "result.csv"))

    def _get_dataset_args(x: str):
        tmp_ls = x.split("_")
        return pd.Series(
            {
                "dataset_raw": "_".join(tmp_ls[:-5]),  # gen_blobs
                "dimension": int(tmp_ls[-5]),
                "n_samples": "_".join(tmp_ls[-4:-2]),
                # "n_samples1": tmp_ls[-4],
                # "n_samples2": tmp_ls[-3],
                "sigma": "_".join(tmp_ls[-2:]),
                "sigma1": float(tmp_ls[-2]),
                "sigma2": float(tmp_ls[-1]),
            }
        )

    # tmpdatasets = pd.Series(df["dataset"].unique())
    # df_ds = tmpdatasets.apply(_get_dataset_args)
    # df = df.merge(df_ds, left_on="dataset", right_index=True)

    df["method"] = df["method"].str.replace("use_weight=", "")

    df = pd.concat([df, df["dataset"].apply(_get_dataset_args)], axis=1)
    return df


def plot_boxplot_single(df: pd.DataFrame, metric: str = "F1", output_dir: str = None):
    for dataset, df_dataset in df.groupby("dataset"):
        # assert df_dataset["n_samples"].nunique() == 1
        dimension = df_dataset["dimension"].unique()[0]
        n_samples = df_dataset["n_samples"].unique()[0]
        fig, ax = plt.subplots(figsize=(10, 10))
        _ = sns.boxplot(data=df_dataset, x="method", y=metric, ax=ax)
        ax.set_title(f"{dataset}")
        if output_dir is not None:
            tmp_output_dir = os.path.join(output_dir, n_samples, str(dimension))
            os.makedirs(tmp_output_dir, exist_ok=True)
            fig.savefig(
                os.path.join(tmp_output_dir, f"{dataset}.png"), bbox_inches="tight"
            )
            plt.close(fig)
        else:
            plt.show()


def plot_boxplot_all(
    df: pd.DataFrame,
    metric: str = "F1",
    output_dir: str = None,
    sigmas: Sequence[float] = None,
    methods: Sequence[str] = None,
):
    if sigmas is not None:
        df = df[df["sigma1"].isin(sigmas) & df["sigma2"].isin(sigmas)]
    if methods is not None:
        df = df[df["method"].isin(methods)]

    for (n_samples, dimension), df_samp_dim in df.groupby(["n_samples", "dimension"]):
        g = sns.FacetGrid(df_samp_dim, row="sigma1", col="sigma2")
        g.map_dataframe(sns.boxplot, x="method", y=metric)
        fig = g.fig

        subplots_ajust_top = 0.95 if df["sigma1"].nunique() > 3 else 0.9
        fig.subplots_adjust(top=subplots_ajust_top)
        fig.suptitle(f"n_samples={n_samples}, dimension={dimension}")
        for ax in g.axes.flat:
            for label in ax.get_xticklabels():
                label.set_rotation(30)
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(
                os.path.join(output_dir, f"{n_samples}-{dimension}.png"),
                bbox_inches="tight",
            )
            plt.close(fig)
        else:
            plt.show()


def exp_estim_pdoc(
    n_neighbors: int = None,
    radius: float = None,
    use_weight=False,
    n_samples: int = 5000,
    sigma: float = 1,
    dimension: int = 2,
    seed: int = 0,
    ret_df: bool = False,
):
    dataset = load_one(
        "gen_blobs",
        {"n_samples": n_samples, "sigma": sigma, "dimension": dimension, "seed": seed},
    )
    df, y_col = dataset.df, dataset.y_col
    X, y = df[df.columns[df.columns != y_col]], df[y_col]
    y = format_y_binary(y)
    # ls_neighbor_idxs = pdoc.get_ls_neighbor_idxs(
    #     X=X, n_neighbors=n_neighbors, radius=radius
    # )
    # pdoc_estim = pdoc.estimate_proba_diff_oppo_cls(
    #     y=y, ls_neighbor_idxs=ls_neighbor_idxs
    # )
    pdoc_estim = estim_pdoc.k_nn(
        X=X, y=y, n_neighbors=n_neighbors, use_weight=use_weight
    )
    pdoc_true = pdoc.calc_pdoc_gaussian(
        X=X,
        y=y,
        pos_proba_prior=0.5,
        neg_mean=[-1] + [0] * (dimension - 1),
        neg_cov=sigma,
        pos_mean=[1] + [0] * (dimension - 1),
        pos_cov=sigma,
    )

    df["pdoc"] = pdoc_true
    df["pdoc_estim"] = pdoc_estim

    # metrics = ["Precision", "Accuracy", "F1", "MSE"]
    metrics = None
    res = pdoc.eval_estimate(pdoc_true, pdoc_estim, metric=metrics)
    # res["exp_id"] = f"{n_neighbors}-{radius}-{use_weight}-{n_samples}-{sigma}-{dimension}-{seed}"
    # res["exp_id_noseed"] = f"{n_neighbors}-{radius}-{use_weight}-{n_samples}-{sigma}-{dimension}"
    res["n_neighbors"] = n_neighbors
    res["radius"] = radius
    res["use_weight"] = use_weight
    res["n_samples"] = n_samples
    res["sigma"] = sigma
    res["dimension"] = dimension
    res["seed"] = seed

    if ret_df:
        return df, res
    else:
        return res


def exp_estim_pdoc2(
    radius: list,
    use_weight=False,
    n_samples=5000,
    sigma=1,
    dimension=2,
    seed=0,
    ret_df=False,
    dir_save=None,
):
    dataset = load_one(
        "gen_blobs",
        {"n_samples": n_samples, "sigma": sigma, "dimension": dimension, "seed": seed},
    )
    df, y_col = dataset.df, dataset.y_col
    X, y = df[df.columns[df.columns != y_col]], df[y_col]

    fig, ax = plt.subplots(figsize=(10, 10))
    df["y"] = df["y"].astype(int).astype("category")
    ax = sns.scatterplot(data=df, x="x1", y="x2", hue=y_col, style=y_col, ax=ax)

    # make the x and y axis equal
    ax.set_aspect("equal", adjustable="box")

    y = format_y_binary(y)
    model_nn = NearestNeighbors()
    model_nn.fit(X.to_numpy())

    # p2draw_x1 = np.linspace(-2.5, 2.5, 6)
    # p2draw_x2 = np.linspace(-3, 3, 7)
    p2draw_x1 = np.linspace(-2, -0.5, 4)
    p2draw_x2 = np.linspace(0, 2, 5)
    p2draw_x1, p2draw_x2 = np.meshgrid(p2draw_x1, p2draw_x2)
    # p2draw_x1 = p2draw_x1.reshape(-1, 1)
    # p2draw_x2 = p2draw_x2.reshape(-1, 1)

    df2draw = pd.DataFrame(
        {
            "x1": p2draw_x1.flatten(),
            "x2": p2draw_x2.flatten(),
        }
    )
    df_tmp = df2draw.copy()
    df2draw["y"] = 1
    df_tmp["y"] = -1
    df2draw = pd.concat([df2draw, df_tmp], axis=0, ignore_index=True)

    y_df2draw = df2draw["y"]
    X_df2draw = df2draw[["x1", "x2"]].to_numpy()
    df2draw["pdoc_true"] = pdoc.calc_pdoc_gaussian(
        X=X_df2draw,
        y=y_df2draw,
        pos_proba_prior=(y_df2draw == 1).sum() / len(y_df2draw),
        neg_mean=[-1] + [0] * (dimension - 1),
        neg_cov=sigma,
        pos_mean=[1] + [0] * (dimension - 1),
        pos_cov=sigma,
    )

    for r in radius:
        ls_neighbor_idxs = model_nn.radius_neighbors(
            X_df2draw, radius=r, return_distance=False
        )
        pdoc_estim = pdoc.estimate_proba_diff_oppo_cls(
            y=y_df2draw, ls_neighbor_idxs=ls_neighbor_idxs, y_train=y
        )
        df2draw[f"pdoc_estim_{r}"] = pdoc_estim
        df2draw[f"MAE_{r}"] = np.abs(df2draw["pdoc_true"] - df2draw[f"pdoc_estim_{r}"])
        df2draw[f"MSE_{r}"] = df2draw[f"MAE_{r}"] ** 2
        df2draw[f"MAPE_{r}"] = df2draw[f"MAE_{r}"] / df2draw["pdoc_true"]

        for i, point in enumerate(X_df2draw):
            # add point
            tmp_artist1 = ax.add_artist(Circle(point, 0.05, color="r", fill=True))

            # add circle
            tmp_artist2 = ax.add_artist(
                Circle(point, r, color="r", fill=False, linewidth=2)
            )

            title = (
                f"p=({point[0]:.2f},{point[1]:.2f})"
                f", true={df2draw['pdoc_true'][i]:.4f}, estim={df2draw[f'pdoc_estim_{r}'][i]:.4f}"
                f", MAE={df2draw[f'MAE_{r}'][i]:.4f}, MSE={df2draw[f'MSE_{r}'][i]:.4f}, MAPE={df2draw[f'MAPE_{r}'][i]:.4f}"
            )
            plt.title(title)
            if dir_save is not None:
                tmpdir = os.path.join(
                    dir_save,
                    str(i),
                )
                os.makedirs(tmpdir, exist_ok=True)
                fig.savefig(
                    os.path.join(tmpdir, f"{point[0]:.2f}_{point[1]:.2f}_{r:.4f}.png"),
                    bbox_inches="tight",
                )
                tmp_artist1.remove()
                tmp_artist2.remove()
            else:
                plt.show()
                fig, ax = plt.subplots(figsize=(10, 10))
                sns.scatterplot(data=df, x="x1", y="x2", hue=y_col, style=y_col, ax=ax)

    return df2draw


def exp_estim_pdoc3(
    radius: list = None,
    n_neighbors: list = None,
    # dir_save=None,
    n_samples=5000,
    sigma=1,
    dimension=2,
    seeds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    distr="blobs",
):
    p2draw_x1 = np.arange(-4, 4.5, 0.5)
    p2draw_x2 = np.arange(0, 3.5, 0.5)
    p2draw_x1, p2draw_x2 = np.meshgrid(p2draw_x1, p2draw_x2)
    df2draw = pd.DataFrame(
        {
            "x1": p2draw_x1.flatten(),
            "x2": p2draw_x2.flatten(),
        }
    )
    df_tmp = df2draw.copy()
    df2draw["y"] = 1
    df_tmp["y"] = -1
    df2draw = pd.concat([df2draw, df_tmp], axis=0, ignore_index=True)
    df2draw["y"] = df2draw["y"].astype(int)

    # Although the type of y is already int, but still need to convert to int
    # since `apply` will change the type of y to float
    df2draw["instance"] = df2draw.apply(
        lambda x: f"({x['x1']:5.2f},{x['x2']:5.2f}),{int(x['y']):2d}", axis=1
    )

    y_df2draw = df2draw["y"]
    X_df2draw = df2draw[["x1", "x2"]].to_numpy()
    if distr == "blobs":
        df2draw["pdoc_true"] = pdoc.calc_pdoc_gaussian(
            X=X_df2draw,
            y=y_df2draw,
            pos_proba_prior=0.5,
            neg_mean=[-1] + [0] * (dimension - 1),
            neg_cov=sigma,
            pos_mean=[1] + [0] * (dimension - 1),
            pos_cov=sigma,
        )
    elif distr == "xor":
        df2draw["pdoc_true"] = pdoc.calc_pdoc_xor(
            X=X_df2draw,
            y=y_df2draw,
            sigma=sigma,
            pos_proba_prior=0.5,
        )
    else:
        raise ValueError

    if radius is not None:
        args = radius
        nn_type = "r_nn"
    elif n_neighbors is not None:
        args = n_neighbors
        nn_type = "k_nn"
    else:
        raise ValueError
    ls_df = []
    df2draw.reset_index(drop=False, inplace=True)
    for seed in seeds:
        if distr == "blobs":
            dataset = load_one(
                "gen_blobs",
                {
                    "n_samples": n_samples,
                    "sigma": sigma,
                    "dimension": dimension,
                    "seed": seed,
                },
            )
        elif distr == "xor":
            dataset = load_one(
                "gen_xor",
                {
                    "n_samples": n_samples,
                    "sigma": sigma,
                    "seed": seed,
                },
            )
        else:
            raise ValueError
        df, y_col = dataset.df, dataset.y_col
        X, y = df[df.columns[df.columns != y_col]], df[y_col]

        y = format_y_binary(y)
        model_nn = NearestNeighbors()
        model_nn.fit(X.to_numpy())

        for arg in args:
            if nn_type == "r_nn":
                radius = arg
                ls_neighbor_idxs = model_nn.radius_neighbors(
                    X_df2draw, radius=radius, return_distance=False
                )
                n_neighbors = [len(ls) for ls in ls_neighbor_idxs]

            elif nn_type == "k_nn":
                n_neighbors = arg
                ls_dists, ls_neighbor_idxs = model_nn.kneighbors(
                    X_df2draw,
                    n_neighbors=n_neighbors,
                )
                radius = [np.max(ls) for ls in ls_dists]
            else:
                raise ValueError
            pdoc_estim = pdoc.estimate_proba_diff_oppo_cls(
                y=y_df2draw, ls_neighbor_idxs=ls_neighbor_idxs, y_train=y
            )
            df2draw_tmp = df2draw.copy()
            df2draw_tmp["radius"] = radius
            df2draw_tmp["n_neighbors"] = n_neighbors
            df2draw_tmp["seed"] = seed
            df2draw_tmp["pdoc_estim"] = pdoc_estim
            df2draw_tmp["MAE"] = np.abs(
                df2draw_tmp["pdoc_true"] - df2draw_tmp["pdoc_estim"]
            )
            df2draw_tmp["MSE"] = df2draw_tmp["MAE"] ** 2
            df2draw_tmp["MAPE"] = df2draw_tmp["MAE"] / df2draw_tmp["pdoc_true"]
            ls_df.append(df2draw_tmp)

    df2draw = pd.concat(ls_df, axis=0)

    return df2draw


def draw_blobs_pdoc(dimension: int = 2, sigma=1, metric="Precision"):
    n_samples = 5000
    seed = 0
    df, res = exp_estim_pdoc(
        n_neighbors=30,
        radius=None,
        n_samples=n_samples,
        sigma=sigma,
        dimension=dimension,
        seed=seed,
        ret_df=True,
    )
    score = res[metric]
    y_col = "y"

    if dimension == 1:
        fig, axes = plt.subplots(2, 3, figsize=(16, 12))
        sns.histplot(data=df, x="x1", hue=y_col, ax=axes[0][0])
        sns.histplot(data=df, x="pdoc", hue=y_col, ax=axes[0][1])
        sns.scatterplot(data=df, x="x1", y="pdoc", hue=y_col, ax=axes[0][2])

        sns.histplot(data=df, x="pdoc_estim", hue=y_col, ax=axes[1][1])
        sns.scatterplot(data=df, x="x1", y="pdoc_estim", hue=y_col, ax=axes[1][2])

        fig.suptitle(
            f"dimension={dimension}, sigma={sigma}, metric={metric}, score={score:.4f}"
        )
        plt.show()

    elif dimension == 2:
        fig, axes = plt.subplots(2, 3, figsize=(16, 12))
        sns.scatterplot(data=df, x="x1", y="x2", hue=y_col, style=y_col, ax=axes[0][0])
        sns.histplot(data=df, x="pdoc", hue=y_col, ax=axes[0][1])
        sns.scatterplot(data=df, x="x1", y="x2", hue="pdoc", style=y_col, ax=axes[0][2])

        sns.histplot(data=df, x="pdoc_estim", hue=y_col, ax=axes[1][1])
        sns.scatterplot(
            data=df, x="x1", y="x2", hue="pdoc_estim", style=y_col, ax=axes[1][2]
        )
        fig.suptitle(
            f"dimension={dimension}, sigma={sigma}, metric={metric}, score={score:.4f}"
        )
        plt.show()
    else:
        raise NotImplementedError


def plot_grid_samples(save_path=None, figsize=None, plot_circle=False, distr="blobs"):
    if distr == "blobs":
        ds = load_one(
            "gen_blobs",
            {
                "n_samples": 5000,
                "sigma": 1,
                "dimension": 2,
                "seed": 0,
            },
        )
    elif distr == "xor":
        ds = load_one(
            "gen_xor",
            {
                "n_samples": 5000,
                "sigma": 1,
                "seed": 0,
            },
        )
    else:
        raise ValueError
    df, y_col = ds.df, ds.y_col
    X, y = df[df.columns[df.columns != y_col]], df[y_col]
    y = format_y_binary(y)

    fig, ax = plt.subplots(figsize=figsize)
    df["y"] = df["y"].astype(int).astype("category")
    palette = None  # TODO: set palette
    ax = sns.scatterplot(
        data=df, x="x1", y="x2", hue=y_col, style=y_col, ax=ax, palette=palette
    )

    # plt.figure()
    # sns.scatterplot(df_rnn[["x1", "x2"]].drop_duplicates(), x="x1", y="x2")
    if distr == "blobs":
        markpoints = np.meshgrid(np.arange(-3, 3.1, 1.5), np.arange(0, 3.1, 1.5))
    elif distr == "xor":
        markpoints = np.meshgrid(np.arange(-3, 3.1, 1.5), np.arange(0, 3.1, 1.5))
        # markpoints = np.meshgrid(np.arange(-1.5, 1.6, 0.5), np.arange(0, 1.6, 0.5))
    else:
        raise ValueError
    # TODO: set size
    sns.scatterplot(
        x=markpoints[0].flatten(), y=markpoints[1].flatten(), color="r", ax=ax
    )

    # plot_circle
    if plot_circle:
        # for x1, x2 in zip(markpoints[0].flatten(), markpoints[1].flatten()):
        for x1, x2 in ((-1.5, 1.5),):
            ax.add_artist(Circle((x1, x2), 1, color="r", fill=False, linewidth=2))

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


def plot_acc_pdoc_rnn(
    save_path=None,
    radius=None,
    figsize=None,
    drawr1r2=True,
    legendoutside=True,
    linestyle=False,
    fontsize=None,
    distr="blobs",
    drawavg=False,
):

    radius = np.arange(0.01, 5.01, 0.01) if radius is None else radius
    df_rnn = exp_estim_pdoc3(radius=radius, distr=distr)
    # the result is the same when y=1 or y=-1 because the pdoc= y * pd
    # only pd_true is different from pd_estim
    if fontsize is not None:
        plt.rcParams.update({"font.size": fontsize})

    fig, ax = plt.subplots(figsize=figsize)
    if distr == "blobs":
        df_rnn_2draw = df_rnn[
            (df_rnn["x1"].isin(np.arange(-3, 3.1, 1.5)))
            & (df_rnn["x2"].isin(np.arange(0, 3.1, 1.5)))
            & (df_rnn["y"] == -1)
            & (df_rnn["radius"].isin(np.arange(0, 5, 0.1)))
        ]
    elif distr == "xor":
        df_rnn_2draw = df_rnn[
            (df_rnn["x1"].isin(np.arange(-3, 3.1, 1.5)))
            & (df_rnn["x2"].isin(np.arange(0, 3.1, 1.5)))
            # (df_rnn["x1"].isin(np.arange(-1.5, 1.6, 0.5)))
            # & (df_rnn["x2"].isin(np.arange(0, 1.6, 0.5)))
            & (df_rnn["y"] == -1)
            & (df_rnn["radius"].isin(np.arange(0, 5, 0.1)))
        ]
    else:
        raise ValueError

    if drawavg:
        dfavg = df_rnn_2draw.groupby(["radius", "seed"])["MSE"].mean().reset_index()
        dfavg["instance"] = "avg"
        df_rnn_2draw = pd.concat([df_rnn_2draw, dfavg], axis=0)

    sns.lineplot(
        df_rnn_2draw,
        x="radius",
        y="MSE",
        hue="instance",
        style="instance" if linestyle else None,
        ax=ax,
    )

    if drawr1r2:
        # draw red lines for r1 r2
        r1 = 1
        r2 = 1.5
        ax.axvline(x=r1, color="r", linestyle="-", label=f"r1={r1:.2f}")
        ax.axvline(x=r2, color="r", linestyle="--", label=f"r2={r2:.2f}")

    if legendoutside:
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()

    if fontsize is not None:
        plt.rcParams.update({"font.size": 10})


def plot_acc_pdoc_rnn_3fig(
    df_rnn: pd.DataFrame,
    save_dir=None,
    figsizes=(None, None, None),
    legendoutside=True,
    linestyle=False,
    fontsize=None,
    distr="blobs",
):

    df_rnn_2draw = df_rnn[
        (df_rnn["x1"].isin(np.arange(-3, 3.1, 1.5)))
        & (df_rnn["x2"].isin(np.arange(0, 3.1, 1.5)))
        & (df_rnn["y"] == -1)
        & (df_rnn["radius"].isin(np.arange(0, 5, 0.1)))
    ]
    tmp_sp = (
        os.path.join(save_dir, f"{distr}_ds_and_samples.png")
        if save_dir is not None
        else None
    )

    if fontsize is not None:
        plt.rcParams.update({"font.size": fontsize})
    plot_grid_samples(
        save_path=tmp_sp, figsize=figsizes[0], plot_circle=True, distr=distr
    )

    fig1, ax1 = plt.subplots(figsize=figsizes[1])
    sns.lineplot(
        df_rnn_2draw,
        x="radius",
        y="MSE",
        hue="instance",
        style="instance" if linestyle else None,
        ax=ax1,
    )
    if legendoutside:
        # Shrink current axis by 20%
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        ax1.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    fig2, ax2 = plt.subplots(figsize=figsizes[2])
    dfavg = df_rnn_2draw.groupby(["radius", "seed"])["MSE"].mean().reset_index()
    sns.lineplot(
        dfavg,
        x="radius",
        y="MSE",
        ax=ax2,
    )

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig1.savefig(
            os.path.join(save_dir, f"{distr}_pdocmse_radius_sample.pdf"),
            bbox_inches="tight",
        )
        fig2.savefig(
            os.path.join(save_dir, f"{distr}_pdocmse_radius_avg.pdf"),
            bbox_inches="tight",
        )
        plt.close("all")
    else:
        plt.show()

    if fontsize is not None:
        plt.rcParams.update({"font.size": 10})


def plot_acc_pdoc_knn(
    save_path=None,
    figsize=None,
    n_neighbors=None,
    drawr1r2=True,
    legendoutside=True,
    linestyle=False,
    fontsize=None,
    distr="blobs",
):
    n_neighbors = list(range(1, 100)) if n_neighbors is None else n_neighbors
    df_knn = exp_estim_pdoc3(n_neighbors=n_neighbors, distr=distr)

    if fontsize is not None:
        plt.rcParams.update({"font.size": fontsize})

    if distr == "blobs":
        df_knn_2draw = (
            df_knn[
                (df_knn["x1"].isin(np.arange(-3, 3.1, 1.5)))
                & (df_knn["x2"].isin(np.arange(0, 3.1, 1.5)))
                & (df_knn["y"] == -1)
            ],
        )
    elif distr == "xor":
        df_knn_2draw = df_knn[
            (df_knn["x1"].isin(np.arange(-3, 3.1, 1.5)))
            & (df_knn["x2"].isin(np.arange(0, 3.1, 1.5)))
            # (df_knn["x1"].isin(np.arange(-1.5, 1.6, 0.5)))
            # & (df_knn["x2"].isin(np.arange(0, 1.6, 0.5)))
            & (df_knn["y"] == -1)
        ]
    else:
        raise ValueError

    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.lineplot(
        df_knn_2draw,
        x="n_neighbors",
        y="MSE",
        hue="instance",
        style="instance" if linestyle else None,
        ax=ax,
    )

    if drawr1r2:
        r1 = 30
        r2 = 60
        ax.axvline(x=r1, color="r", linestyle="-", label=f"r1=R{r1}")
        ax.axvline(x=r2, color="r", linestyle="--", label=f"r2=R{r2}")

    if legendoutside:
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()

    if fontsize is not None:
        plt.rcParams.update({"font.size": 10})


def plot_acc_pdoc_knn_avg(
    df_knn: pd.DataFrame,
    save_path=None,
    figsize=None,
    drawr1r2=True,
    fontsize=None,
):

    if fontsize is not None:
        plt.rcParams.update({"font.size": fontsize})

    df_knn_2draw = df_knn[
        (df_knn["x1"].isin(np.arange(-3, 3.1, 1.5)))
        & (df_knn["x2"].isin(np.arange(0, 3.1, 1.5)))
        & (df_knn["y"] == -1)
    ]
    df_knn_avg = (
        df_knn_2draw.groupby(["n_neighbors", "seed"])["MSE"].mean().reset_index()
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.lineplot(
        df_knn_avg,
        x="n_neighbors",
        y="MSE",
        ax=ax,
    )

    if drawr1r2:
        r1 = 30
        r2 = 60
        ax.axvline(x=r1, color="r", linestyle="-", label=f"r1=R{r1}")
        ax.axvline(x=r2, color="r", linestyle="--", label=f"r2=R{r2}")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()

    if fontsize is not None:
        plt.rcParams.update({"font.size": 10})


def main():

    for distr in ("blobs", "xor"):
        df_rnn = exp_estim_pdoc3(radius=np.arange(0.01, 5.01, 0.01), distr=distr)
        plot_acc_pdoc_rnn_3fig(
            df_rnn=df_rnn,
            distr=distr,
            save_dir=r"../output/analysis_pdoc",
            fontsize=16,
            figsizes=((6, 6), (8, 6), (6, 6)),
        )

        df_knn = exp_estim_pdoc3(n_neighbors=list(range(1, 100)), distr=distr)

        plot_acc_pdoc_knn_avg(
            df_knn=df_knn,
            drawr1r2=False,
            fontsize=18,
            save_path=f"../output/analysis_pdoc/{distr}_pdocmse_nn_avg.pdf",
        )


def debug():
    # df = read_and_preprocess(r"..\output\result\estim_pdoc\20230721")
    # draw_blobs_pdoc()
    args = itertools.product(range(2, 100, 2), [True, False], range(2))
    ls_res = joblib.Parallel(n_jobs=8, verbose=1)(
        joblib.delayed(exp_estim_pdoc)(
            n_neighbors=n_neighbors,
            radius=None,
            use_weight=use_weight,
            n_samples=5000,
            sigma=1,
            dimension=2,
            seed=seed,
            ret_df=False,
        )
        for n_neighbors, use_weight, seed in args
    )

    df_res = pd.DataFrame(ls_res)

    # it seems that the use_weight=False is better than use_weight=True
    sns.lineplot(
        df_res,
        x="n_neighbors",
        y="AUC",
        hue="use_weight",
    )


if __name__ == "__main__":
    main()
    # debug()
