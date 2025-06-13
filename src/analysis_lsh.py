import itertools
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from ismethod import lshqiu


def gen_uniform(n_samples: int = 5000, dimension: int = 2, seed: int = 1234):
    """Generate uniform random samples in a unit hypercube."""
    rng = np.random.default_rng(seed)
    return rng.uniform(size=(n_samples, dimension))


def gen_normal(n_samples: int = 5000, dimension: int = 2, seed: int = 1234):
    """Generate normal random samples."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_samples, dimension))


def exp(data_name, n_samples, dimension, seed):
    if data_name == "uniform":
        data = gen_uniform(n_samples, dimension, seed)
    elif data_name == "normal":
        data = gen_normal(n_samples, dimension, seed)
    else:
        raise ValueError(f"Unknown data_name: {data_name}")
    data = StandardScaler().fit_transform(data)
    param_w, param_t, param_L = lshqiu.get_lsh_ours_op_args(data, seed=seed)
    return data_name, n_samples, dimension, seed, param_w, param_t, param_L


def exp_multi(
    data_names=("uniform", "normal"),
    lst_n_samples=(1000, 2000, 5000, 10000, 20000),
    dimensions=(2, 3, 5, 10, 20),
    seeds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
):
    args = itertools.product(data_names, lst_n_samples, dimensions, seeds)
    results = joblib.Parallel(n_jobs=-1)(joblib.delayed(exp)(*arg) for arg in args)
    df_res = pd.DataFrame(
        results,
        columns=["data_name", "n_samples", "dimension", "seed", "w", "t", "L"],
    )
    return df_res


def draw_heatmap(df_res: pd.DataFrame, fontsize=None, save_dir: str = None):
    if fontsize is not None:
        plt.rcParams.update({"font.size": fontsize})

    for data_name, df in df_res.groupby("data_name"):
        for v in ["t", "L", "w"]:
            fig, ax = plt.subplots()
            ax = sns.heatmap(
                df.pivot_table(
                    index="dimension", columns="n_samples", values=v
                ).sort_index(ascending=False),
                annot=True,
                fmt=".1f",
                ax=ax,
                cbar_kws={"format": "%.1f"},
            )
            ax.set_xlabel("Number of samples")
            ax.set_ylabel("Dimension")

            title = f"{data_name}_{v}"
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                fig.savefig(os.path.join(save_dir, f"{title}.pdf"), bbox_inches="tight")
            else:
                ax.set_title(title)
                plt.show()
            plt.close(fig)

    if fontsize is not None:
        plt.rcParams.update({"font.size": 10})


def main():

    df_res = exp_multi()
    draw_heatmap(df_res, fontsize=16, save_dir="../output/analysis_lsh/")
    # df = df_res[(df_res["data_name"] == "uniform") & (df_res["dimension"] == 2)]
    # fig, ax = plt.subplots()
    # sns.lineplot(data=df, x="n_samples", y="t", ax=ax)


if __name__ == "__main__":
    main()
