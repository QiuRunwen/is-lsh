"""Analysis the experiment results that produced by `exp.py`."""

import shutil
import os
import warnings
from types import MappingProxyType

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

sns.reset_defaults()
RESULT_DIR_SYN = "../output/result/current/synthetic_ds"
RESULT_DIR_NAT = "../output/result/current/natural_ds"

# TODO DATASET
DIR_DATASET_RENAME = MappingProxyType(
    {
        "load_mammographic": "Mammographic",
        "load_obesity": "Obesity",
        "load_mushroom": "Mushroom",
        "load_nursery": "Nursery",
        "load_banknote": "Banknote",
        "load_smsspam": "SMS",
        "load_twonorm": "TwoNorm",
        "load_banana": "Banana",
        "gen_blobs": "Blobs",
        "gen_moons": "Moons",
        "gen_xor": "XOR",
        "gen_circles": "Circles",
    }
)

DATASETS_NAT = (
    "Mammographic",
    "Obesity",
    "Mushroom",
    "Nursery",
    "Banknote",
    "SMS",
)

DATASETS_SYN = (
    "Blobs",
    "Moons",
    "XOR",
    "Circles",
    "Banana",
    "TwoNorm",
)


DIR_METHOD_RENAME = MappingProxyType(
    {
        "lsh_ours": "PDOC-LSH",
        "lsh_dr": "DR-LSH",
        "lsh_bp": "BP-LSH",
        "LSH_IS_F_bs": "LSH-F-BS",
        "ENNC": "ENNC",
        "NC": "NC",
        "noselect": "noselect",
        "PDOC_Voronoi": "PDOC-V",
    }
)
METHODS = (
    "noselect",
    "PDOC-LSH",
    "DR-LSH",
    "BP-LSH",
    "LSH-F-BS",
    "ENNC",
    "NC",
    "NE",
    "PDOC-V",
)

DIR_MODEL_RENAME = MappingProxyType(
    {
        "MLPClassifier": "MLP",
        "GradientBoostingClassifier": "GBDT",
        "RandomForestClassifier": "RF",
        "LogisticRegression": "LR",
        "KNeighborsClassifier": "KNN",
        "SVC": "SVM",
    }
)
MODELS = (
    "LR",
    "SVM",
    "MLP",
    "RF",
    "GBDT",
    "KNN",
)

METRICS = (
    "Accuracy",
    "F1",
    "AUC",
    "Precision",
    "Recall",
    "time",
    "time_method",
    "time_model",
)


IS_DATA_SYN = False
if IS_DATA_SYN:
    DATASETS = DATASETS_SYN
    RESULT_DIR = RESULT_DIR_SYN
else:
    DATASETS = DATASETS_NAT
    RESULT_DIR = RESULT_DIR_NAT


def find_all_path(result_dir, file_fullname="result.csv"):
    """Find all the `file_fullname` files in the subdirectories."""
    filepath_ls = []
    filedir_ls = []
    for root, dirs, files in os.walk(result_dir):
        if file_fullname in files:
            filepath_ls.append(os.path.join(root, file_fullname))
            filedir_ls.append(root)
    return filepath_ls, filedir_ls


def make_str_right_type(s: str):
    """Convert string to bool, float or keep string"""

    # check if is bool
    s_lower = s.lower()
    if s_lower == "true":
        return True
    if s_lower == "false":
        return False

    # check if is float
    try:
        return float(s)
    except ValueError:
        return s


def clean_result(
    result_dir: str,
    bakup=True,
    result_name="result.csv",
    method2del="NC",
    verbose=True,
):
    """Clean the result from the `result_dir`, remove the method2del."""

    msg = "--- clean_result ---"
    msg += f"\nClean result from {result_dir}"
    if bakup:
        # backup the result_dir
        bakup_dir = f"{result_dir}_bak"
        shutil.copytree(result_dir, bakup_dir)

    filepath_ls, _ = find_all_path(result_dir, file_fullname=result_name)
    for filepath in filepath_ls:
        df = pd.read_csv(filepath)
        nrows_before = df.shape[0]
        tmp = df["method"].transform(lambda x: x.split("_")[0])
        # df = df[~df["method"].str.contains(method2del)]
        df = df[tmp != method2del]
        nrows_after = df.shape[0]
        msg += f"\n{filepath}: drop {nrows_before-nrows_after} rows with method {method2del}"
        df.to_csv(filepath, index=False)

    msg += "\n--- end of clean_result ---"
    if verbose:
        print(msg)


def read_result(
    result_dir: str,
    result_name="result.csv",
    all_csv=False,
    duplicated_subset: list = None,
    verbose=True,
):
    """Read the result from the `result_dir`."""
    msg = "--- read_result ---"
    msg += f"\nRead result from {result_dir}"
    if all_csv:
        filepath_ls, _ = find_all_path(result_dir, file_fullname=result_name)
        dfs = map(pd.read_csv, filepath_ls)
        df = pd.concat(dfs, axis=0)
        msg += f"\nconcat {len(filepath_ls)} csv files"
    else:
        df = pd.read_csv(os.path.join(result_dir, "result.csv"))

    if duplicated_subset is None:
        duplicated_subset = ["exp_id"]

    if len(duplicated_subset) != 0:
        nrows_before = df.shape[0]
        df.drop_duplicates(subset=duplicated_subset, inplace=True)
        nrows_after = df.shape[0]
        msg += (
            f"\ndrop {nrows_before-nrows_after} duplicated rows by {duplicated_subset}"
        )

    msg += f"\nshape={df.shape}"
    msg += "\n--- end of read_result ---"

    if verbose:
        print(msg)

    return df


def _f_extract_param(x: str, d_pattern_len: dict, ser: pd.Series):
    ser = ser.copy()
    if x == "noselect":
        ser["method"] = "noselect"
        ser["selection_rate"] = 1
    else:
        tmpls = x.split("_")
        max_index = len(tmpls) - 1
        curr_i = 0
        for name, length in d_pattern_len.items():
            # d[name] = make_str_right_type("_".join(tmpls[curr_i : curr_i + length]))
            ser[name] = "_".join(tmpls[curr_i : curr_i + length])
            curr_i += length
            if curr_i > max_index:
                break
        if curr_i <= max_index:
            warnings.warn(f"the length of {x} is longer than expected.")
    return ser


def preprocess_result(
    df: pd.DataFrame,
    dir_dataset_rename=DIR_DATASET_RENAME,
    dir_method_rename=DIR_METHOD_RENAME,
    dir_model_rename=DIR_MODEL_RENAME,
    dss=DATASETS,
    methods=METHODS,
    models=MODELS,
    metrics2rel: list[str] = None,
    d_col_pattern=None,
    verbose=True,
):
    """Preprocess the result dataframe."""
    msg = "--- preprocess_result ---"

    df["time"] = df["time_method"] + df["time_model"]
    if d_col_pattern is None:
        d_col_pattern = {
            "method": {"method": (2, str), "selection_rate": (1, float)},
            "dataset": {
                "dataset": (2, str),
                "n_sample": (1, float),
                "sigma": (1, float),
                "dimension": (1, float),
            },
            # "model": {"model": (1, str)},
        }

    for col, d_pattern_tmp in d_col_pattern.items():
        # TODO
        # methods that have diff len will be wrong. such as "LSH_IS_F_bs", "ENNC", "lsh_dp"
        d_pattern_len = {k: v[0] for k, v in d_pattern_tmp.items()}
        d_pattern_type = {
            k: v[1] if v[1] not in (str, "str") else "object"
            for k, v in d_pattern_tmp.items()
        }

        if len(d_pattern_len) == 1:
            msg += f"\nno need to extract column `{col}`"
        elif col == "method" and len(d_pattern_len) == 2:
            df["method_old"] = df["method"].copy()
            df["selection_rate"] = df["method"].transform(
                lambda x: float(x[-4:]) if x != "noselect" else 1
            )
            df["method"] = df["method"].transform(
                lambda x: x[:-5] if x != "noselect" else "noselect"
            )
            msg += f"\nextract column `{col}` to `method` and `selection_rate`"
        else:
            if (
                col == "method"
                and len(d_pattern_len) > 2
                and df["method"].str.split(r"_", expand=True).shape[1] < 3
            ):
                warnings.warn(
                    "the column `method` does not contain the expected number of `_`"
                )

            ser = pd.Series(np.NaN, index=d_pattern_len.keys(), dtype=object)
            df_tmp = df[col].apply(
                _f_extract_param,
                d_pattern_len=d_pattern_len,
                ser=ser,
            )
            df_tmp = df_tmp.astype(d_pattern_type)

            cols2reanme = []
            for _col in df_tmp.columns:
                if _col in df.columns:
                    cols2reanme.append(_col)
            df.rename(columns={col: f"{col}_old" for col in cols2reanme}, inplace=True)
            df = pd.concat([df, df_tmp], axis=1)
            msg += f"\nextract column `{col}` to `{df_tmp.columns.to_list()}`"

    # rename the dataset, method, model
    df["dataset"] = df["dataset"].transform(lambda x: dir_dataset_rename.get(x, x))
    df["method"] = df["method"].transform(lambda x: dir_method_rename.get(x, x))
    df["model"] = df["model"].transform(lambda x: dir_model_rename.get(x, x))

    dss = df["dataset"].unique() if dss is None else dss
    methods = df["method"].unique() if methods is None else methods
    models = df["model"].unique() if models is None else models
    df = df[
        df["dataset"].isin(dss) & df["method"].isin(methods) & df["model"].isin(models)
    ]

    if verbose:
        print(msg)

    if metrics2rel:
        scenario_cols = ["seed"]
        scenario_cols.append(
            "dataset_old" if "dataset_old" in df.columns else "dataset"
        )
        scenario_cols.append("model_old" if "model_old" in df.columns else "model")
        df = add_relative_performance(
            df, metrics=metrics2rel, scenario_cols=scenario_cols, verbose=verbose
        )
        msg += "\nadd relative performance"

    return df


def add_relative_performance(
    df: pd.DataFrame,
    metrics=METRICS,
    scenario_cols=("seed", "dataset", "model"),
    baseline_method="noselect",
    verbose=True,
):
    """Add relative performance of methods in each scenario."""

    def func(df_tmp: pd.DataFrame):
        for metric in metrics:
            ser_noselect = df_tmp[df_tmp["method"] == baseline_method][metric]
            noselect_size = ser_noselect.shape[0]
            if noselect_size == 1:
                df_tmp[f"relative_{metric}"] = df_tmp[metric] / ser_noselect.iloc[0]
            elif noselect_size == 0:
                warnings.warn(f"no {baseline_method} in the group")
                df_tmp[f"relative_{metric}"] = np.nan
            else:
                raise ValueError(f"more than one {baseline_method} in the group")

        return df_tmp

    if verbose:
        print(
            f"Add relative performance for `{metrics}` to the dataframe"
            f" using `{baseline_method}` as the baseline and group by `{scenario_cols}`"
        )

    return (
        df.groupby(list(scenario_cols), as_index=False)
        .apply(func)
        .reset_index(drop=True)
    )


def plot_pdoc_heatmap(
    df: pd.DataFrame,
    metric: str = "F1",
    use_relative: bool = True,
    output_dir: str = None,
):
    """Plot the heatmap of the pdoc method."""
    df = add_relative_performance(df, [metric])

    if use_relative:
        metric = f"relative_{metric}"
    for dataset, df_dataset in df.groupby("dataset"):
        for model, df_dataset_model in df_dataset.groupby("model"):
            performance_noselect = df_dataset_model[
                df_dataset_model["method"] == "noselect"
            ][metric].mean()

            df_pdoc: pd.DataFrame = df_dataset_model[
                df_dataset_model["method"].str.startswith("pdoc_distr")
            ].copy()

            # def extract_alpha_threshold(method: str):
            #     tmp_ls = method.split("_")
            #     return pd.Series(
            #         {
            #             # "threshold1": str(tmp_ls[-4:-2]),
            #             # "threshold2": str(tmp_ls[-2:]),
            #             "select_rate": tmp_ls[-5],
            #             "threshold1": float(tmp_ls[-4]),
            #             "threshold2": float(tmp_ls[-2]),
            #         }
            #     )

            # df_pdoc = pd.concat(
            #     [df_pdoc, df_pdoc["method"].apply(extract_alpha_threshold)], axis=1
            # )

            df_pdoc_method_para = df_pdoc["method"].str.split("_", expand=True)
            df_pdoc["select_rate"] = df_pdoc_method_para.iloc[:, -5]
            df_pdoc["threshold1"] = df_pdoc_method_para.iloc[:, -4].astype(float)
            df_pdoc["threshold2"] = df_pdoc_method_para.iloc[:, -2].astype(float)

            n_select_rate = df_pdoc["select_rate"].nunique()
            fig, axes = plt.subplots(1, n_select_rate, figsize=(10 * n_select_rate, 10))
            if n_select_rate == 1:
                axes = [axes]
            for ax, (select_rate, df_tmp) in zip(axes, df_pdoc.groupby("select_rate")):
                df_cross = df_tmp.pivot_table(
                    index="threshold1",
                    columns="threshold2",
                    values=metric,
                    aggfunc="mean",
                )

                # for symmetrical
                for thrsh in df_cross.index:
                    if thrsh not in df_cross.columns:
                        df_cross[thrsh] = np.nan
                df_cross.sort_index(axis=0, inplace=True)
                df_cross.sort_index(axis=1, inplace=True)

                sns.heatmap(df_cross, annot=True, ax=ax)
                title = f"{dataset}, {select_rate}, {model}, {metric}, noselect={performance_noselect:.2f}"
                ax.set_title(title)

            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                file_name = f"{dataset}-{model}-{metric}.png"
                fig.savefig(os.path.join(output_dir, file_name), bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()


def plot_pdoc_heatmap_dir(
    result_dir: str,
    metric: str = "F1",
    use_relative: bool = True,
    output_dir: str = None,
):
    # find out all the `result.csv` files in the subdirectories
    filepath_ls, filedir_ls = find_all_path(result_dir, file_fullname="result.csv")

    for filepath, filedir in zip(filepath_ls, filedir_ls):
        df = pd.read_csv(filepath)
        if output_dir is None:
            tmp_output_dir = os.path.join(filedir, "heatmap")
        else:
            tmp_output_dir = output_dir

        plot_pdoc_heatmap(
            df, metric=metric, use_relative=use_relative, output_dir=tmp_output_dir
        )
        print(f"Finished plotting heatmap for {filedir}")


def analyze_delete_misclf(result_dir: str, metric: str, output_dir: str = None):
    df = read_result(result_dir, all_csv=True, duplicated_subset=["exp_id"])
    df = add_relative_performance(df, [metric])

    df_method_para = df["method"].str.split("_", expand=True)
    df["selection_rate"] = df_method_para[2].replace("None", np.nan).astype(float)
    df["threshold1"] = df_method_para[3].replace("None", np.nan).astype(float)
    df["threshold2"] = df_method_para[5].replace("None", np.nan).astype(float)
    df_ds_para = df["dataset"].str.split("_", expand=True)
    df["dimension"] = df_ds_para[2].replace("None", np.nan).astype(int)
    df["sample1"] = df_ds_para[3].replace("None", np.nan).astype(int)
    df["sample2"] = df_ds_para[4].replace("None", np.nan).astype(int)
    df["sigma1"] = df_ds_para[5].replace("None", np.nan).astype(float)
    df["sigma2"] = df_ds_para[6].replace("None", np.nan).astype(float)

    df_mean = (
        df.drop(columns=["exp_id", "seed"])
        .groupby(["dataset", "method", "model"], as_index=False)
        .mean()
    )
    for model, df_model in df_mean.groupby("model", as_index=False):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, (thresholds, df_model_threshold) in zip(
            axes, df_model.groupby(["threshold1", "threshold2"])
        ):
            df_tmp = df_model_threshold[
                (df_model_threshold["method"] != "noselect")
                & df_model_threshold["selection_rate"].isna()
            ]
            sns.scatterplot(
                data=df_tmp, x="sample1", y=f"relative_{metric}", hue="sigma1", ax=ax
            )
            count_greater = (df_tmp[f"relative_{metric}"] > 1).sum()
            ax.set_title(f"{model}, {thresholds}, {count_greater}/{df_tmp.shape[0]}")
            ax.axhline(y=1, color="red", linestyle="--")

        if output_dir is not None:
            detailed_output_dir = os.path.join(output_dir, "detailed")
            os.makedirs(detailed_output_dir, exist_ok=True)
            file_name = f"{model}-{metric}.png"
            fig.savefig(
                os.path.join(detailed_output_dir, file_name), bbox_inches="tight"
            )
            plt.close(fig)

            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            df_tmp = df_model[
                (df_model["method"] != "noselect") & df_model["selection_rate"].isna()
            ].copy()
            df_tmp = df_tmp[
                (df_tmp["threshold1"] == 0) & (df_tmp["threshold2"] == 0)
            ]  # the situation in which only keep the right points
            df_tmp["sample1/all"] = df_tmp["sample1"] / (
                df_tmp["sample1"] + df_tmp["sample2"]
            )
            df_tmp["n_pos/all"] = 1 - df_tmp["sample1/all"]
            sns.scatterplot(
                data=df_tmp,
                x="n_pos/all",
                y=f"relative_{metric}",
                hue="sigma1",
                ax=ax,
            )
            ax.set_title(f"{model}")
            # count_greater = (df_tmp[f"relative_{metric}"] > 1).sum()
            # ax.set_title(f"{model}, {count_greater}/{df_tmp.shape[0]}")
            ax.axhline(y=1, color="red", linestyle="--")
            file_name = f"{model}-{metric}-presentation.png"
            presentation_output_dir = os.path.join(output_dir, "presentation")
            os.makedirs(presentation_output_dir, exist_ok=True)
            fig.savefig(
                os.path.join(presentation_output_dir, file_name), bbox_inches="tight"
            )
            plt.close(fig)
        else:
            plt.show()


def draw_line_perf(
    df: pd.DataFrame,
    methods=None,
    metric="F1",
    save_dir=None,
    verbose=False,
    pdf=False,
    title=True,
    fontsize=None,
):

    methods = list(METHODS) if methods is None else methods
    if fontsize is not None:
        plt.rcParams.update({"font.size": fontsize})

    fig, ax = plt.subplots()
    if verbose:
        total_num_fig = df["model"].nunique() * df["dataset"].nunique()
        progress_bar = tqdm(total=total_num_fig, desc="draw_line_perf")
    for model, df_model in df.groupby("model"):
        if save_dir is not None:
            save_dir_m = os.path.join(save_dir, model)
            os.makedirs(save_dir_m, exist_ok=True)
        for ds, df_model_ds in df_model.groupby("dataset"):
            baseline = df_model_ds[df_model_ds["method"] == "noselect"][metric].mean()
            df2plot = df_model_ds[df_model_ds["method"] != "noselect"]
            method_order = [m for m in methods if m != "noselect"]
            # d_method_order = {m: i for i, m in enumerate(method_order)}
            # df2plot = df2plot.sort_values("method", key=lambda x: x.map(d_method_order), inplace=False)
            if df2plot.empty:
                continue
            if baseline is not np.nan:
                ax.axhline(y=baseline, color="red", linestyle="--", label="noselect")
            ax = sns.lineplot(
                df2plot,
                x="selection_rate",
                y=metric,
                hue="method",
                hue_order=method_order,
                style="method",
                style_order=method_order,
                # sizmarkersize=10,
                markers=True,
                dashes=False,
                ax=ax,
            )
            if title:
                ax.set_title(f"{model}, {ds}")
            if metric == "time":
                ax.set_yscale("log")

            # from matplotlib.ticker import FormatStrFormatter, FuncFormatter
            # ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.3f}"[1:] if x!=1 else "1"))

            if save_dir is not None:
                ext = "pdf" if pdf else "png"
                file_name = f"{model}-{ds}.{ext}"
                fig.savefig(os.path.join(save_dir_m, file_name), bbox_inches="tight")
                ax.clear()
            else:
                plt.show()
                fig, ax = plt.subplots()

            if verbose:
                progress_bar.update(1)

    if fontsize is not None:
        plt.rcParams.update({"font.size": 10})

    plt.close(fig)


def table_perf(
    df: pd.DataFrame, methods=None, metric="F1", save_dir=None, verbose=False
):

    methods = list(METHODS) if methods is None else methods
    if verbose:
        total_num_table = df["model"].nunique() * df["dataset"].nunique()
        progress_bar = tqdm(total=total_num_table, desc="table_perf")
    for model, df_model in df.groupby("model"):
        if save_dir is not None:
            save_dir_m = os.path.join(save_dir, model)
            os.makedirs(save_dir_m, exist_ok=True)
        for ds, df_model_ds in df_model.groupby("dataset"):
            df_pivot_mean = df_model_ds.pivot_table(
                values=metric,
                index="selection_rate",
                columns="method",
                aggfunc="mean",
                dropna=False,
            )
            df_pivot_std = df_model_ds.pivot_table(
                values=metric,
                index="selection_rate",
                columns="method",
                aggfunc="std",
                dropna=False,
            )

            method_order = [m for m in methods if m in df_pivot_mean.columns]
            df_pivot_mean = df_pivot_mean[method_order]
            df_pivot_std = df_pivot_std[method_order]

            df_pivot = (
                df_pivot_mean.map(lambda x: f"{x:.3f}")
                + "±"
                + df_pivot_std.map(lambda x: f"{x:.3f}")
            )
            if save_dir is not None:
                # df_pivot_mean.to_csv(os.path.join(save_dir_m, f"{model}-{ds}-mean.csv"))
                # df_pivot_std.to_csv(os.path.join(save_dir_m, f"{model}-{ds}-std.csv"))
                df_pivot.to_csv(os.path.join(save_dir_m, f"{model}-{ds}.csv"))
            else:
                print(df_pivot)

            if verbose:
                progress_bar.update(1)


def table_selrate2achieve_relperf(
    df: pd.DataFrame,
    metric="F1",
    save_path=None,
    relperf_thresholds=(0.95, 0.98, 0.99),
):
    """Table of selection rate of the method to achieve relative performance under different scenarios."""
    metric_rel = f"relative_{metric}"

    df = df[df["method"] != "noselect"]
    df_mean = df.groupby(
        ["dataset", "method", "model", "selection_rate"], as_index=False
    )[metric_rel].mean()

    dfs = []
    for threshold in relperf_thresholds:
        df_above = df_mean[df_mean[metric_rel] >= threshold]
        if df_above.empty:
            continue

        df_min_selrate = df_above.groupby(
            ["dataset", "method", "model"], as_index=False
        )["selection_rate"].min()

        df_min_selrate_agg = df_min_selrate.groupby(
            ["method", "model"], as_index=False
        )["selection_rate"].agg(["mean", "count"])
        df_min_selrate_agg["threshold"] = threshold
        dfs.append(df_min_selrate_agg)

    df_res: pd.DataFrame = pd.concat(dfs, axis=0)
    if df_res.empty:
        raise ValueError("No method can achieve the relative performance.")

    df_res["avg_selrate(count)"] = (
        df_res["mean"].map(lambda x: f"{x:.3f}") + "(" + df_res["count"].map(str) + ")"
    )
    df_pivot = df_res.pivot(
        values="avg_selrate(count)", index=["threshold", "model"], columns="method"
    )

    # # sort TODO
    # models_unique = df_res["model"].unique()
    # methods_unique = df_res["method"].unique()
    # models = [m for m in MODELS if m in models_unique]
    # methods = [m for m in METHODS if m in methods_unique]
    # df_pivot: pd.DataFrame = df_pivot.loc[(relperf_thresholds, models), methods]

    if save_path is not None:
        df_pivot.to_csv(save_path)

    return df_pivot


def format_table_selrate2achieve_relperf(df: pd.DataFrame, save_path=None):
    """Format the table of selection rate of the method to achieve relative performance under different scenarios."""
    # df_avg_minselrate = df.applymap(lambda x: float(x.split("(")[0]))
    # df_count_above = df.applymap(lambda x: int(x.split("(")[1][:-1]))

    # for each row, find the method with largest count and the smaller selection rate

    def func_row_formatcol(row: pd.Series):
        ser_method_selrate = row.map(
            lambda x: float(x.split("(")[0]) if isinstance(x, str) else x
        )
        ser_method_count = row.map(
            lambda x: int(x.split("(")[1][:-1]) if isinstance(x, str) else x
        )

        bestselrate = ser_method_selrate.min()
        bestcount = ser_method_count.max()

        all_bestmethod4selrate = ser_method_selrate[
            ser_method_selrate == bestselrate
        ].index
        all_bestmethod4count = ser_method_count[ser_method_count == bestcount].index
        ser_method_style = pd.Series("", index=row.index)

        # bold the best method for count
        ser_method_style[all_bestmethod4count] = "font-weight: bold"

        # underline the best method for selection rate
        ser_method_style[all_bestmethod4selrate] += "; text-decoration: underline"

        return ser_method_style

    df_style = df.style.apply(func_row_formatcol, axis=1)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_style.to_excel(save_path, engine="openpyxl")
    return df_style


def analyze_ours_param():

    # for most cases in natural_ds and synthetic_ds
    # use_weight=True is slightly better than use_weight=False, especially when selection_rate is small.
    # it means that when calculating pdoc, using collision count as the weight of neighbors
    # is better than not using it.
    # random is better than filter.
    # near is better than far.
    # near_random_True should be the best.
    result_dir = r"../output/result/lsh_ours_param"
    df = read_result(result_dir, all_csv=True, duplicated_subset=["exp_id"])
    df = preprocess_result(
        df=df,
        dss=None,
        methods=None,
        models=None,
        d_col_pattern={
            "method": {
                "method": (2, str),
                "selection_type": (2, str),
                "selection_rate": (1, float),
                "use_weight": (1, str),  # TODO bool can't be extracted
                "threshold": (1, str),
            },
            "dataset": {
                "dataset": (2, str),
                "n_sample": (1, float),
                "sigma": (1, float),
                "dimension": (1, float),
            },
        },
        metrics2rel=["F1"],
    )

    # df["threshold1"] = df["threshold"].transform(lambda x: float(x.split(",")[0][1:]) if x else 0)
    # df["threshold2"] = df["threshold"].transform(lambda x: float(x.split(",")[0][1:]) if x else 1)

    tmpfilter = df["dataset"].isin(DATASETS_SYN)
    dfsyn = df[tmpfilter].copy()
    dfnat = df[~tmpfilter].copy()

    dfnat["method"] = dfnat["selection_type"] + dfnat["use_weight"] + df["threshold"]
    dfnat["method"].fillna("noselect", inplace=True)
    # draw_line_perf(
    #     dfnat,
    #     methods=dfnat["method"].unique(),
    #     save_dir=os.path.join(result_dir, "line_perf", "F1", "nat", "all"),
    # )
    # table_perf(
    #     dfnat,
    #     methods=dfnat["method"].unique(),
    #     save_dir=os.path.join(result_dir, "table_perf", "F1", "nat", "all"),
    # )

    # dfnat["method"] = dfnat["selection_type"]
    dfnat["method"] = dfnat["selection_type"] + df["threshold"]
    dfnat["method"].fillna("noselect", inplace=True)
    for use_weight, df_use_weight in dfnat.groupby("use_weight"):
        dftmp = dfnat[dfnat["method"] == "noselect"]
        df_use_weight = pd.concat([df_use_weight, dftmp], axis=0)
        draw_line_perf(
            df_use_weight,
            methods=dfnat["method"].unique(),
            save_dir=os.path.join(
                result_dir, "line_perf", "F1", "nat", f"use_weight={use_weight}"
            ),
        )
        table_perf(
            df_use_weight,
            methods=dfnat["method"].unique(),
            save_dir=os.path.join(
                result_dir, "table_perf", "F1", "nat", f"use_weight={use_weight}"
            ),
        )

    dfsyn["method"] = dfsyn["selection_type"] + dfsyn["use_weight"] + df["threshold"]
    dfsyn["method"].fillna("noselect", inplace=True)
    dfsyn["n_sample"] = dfsyn["n_sample"].astype(int)
    dfsyn["dimension"].fillna(2, inplace=True)
    dfsyn["dimension"] = dfsyn["dimension"].astype(int)
    total_num = (
        dfsyn["n_sample"].nunique()
        * dfsyn["sigma"].nunique()
        * dfsyn["dimension"].nunique()
    )
    progress_bar = tqdm(total=total_num, desc="analyze_ours_param")
    for n_sample, df_sample in dfsyn.groupby("n_sample"):
        for sigma, df_sample_sigma in df_sample.groupby("sigma"):
            for dimension, df_sample_sigma_dim in df_sample_sigma.groupby("dimension"):
                draw_line_perf(
                    df_sample_sigma_dim,
                    methods=dfnat["method"].unique(),
                    save_dir=os.path.join(
                        result_dir,
                        "line_perf",
                        "F1",
                        "syn",
                        f"{n_sample}-{sigma:.2f}-{dimension}",
                        "all",
                    ),
                )
                table_perf(
                    df_sample_sigma_dim,
                    methods=dfnat["method"].unique(),
                    save_dir=os.path.join(
                        result_dir,
                        "table_perf",
                        "F1",
                        "syn",
                        f"{n_sample}-{sigma:.2f}-{dimension}",
                        "all",
                    ),
                )

                progress_bar.update(1)


def analyze_pdoc_param():

    ##### Synthetic Datasets
    # "near_filter" and "far_filter" sometimes are very poor when selection_rate is small
    # "near_random", "far_random" and "all_random" are similar
    result_dir = r"../output/result/pdoc_inblob"
    df = read_result(result_dir, all_csv=True, duplicated_subset=["exp_id"])
    df = preprocess_result(
        df,
        dss=None,
        models=None,
        methods=None,
        d_col_pattern={
            "method": {
                "method": (2, str),
                "selection_type": (2, str),
                "selection_rate": (1, float),
            },
            "dataset": {
                "dataset": (2, str),
                "n_sample": (1, float),
                "sigma": (1, float),
                "dimension": (1, float),
            },
        },
        metrics2rel=["F1"],
    )
    df["n_sample"] = df["raw_train_size"]
    df["dimension"] = df["train_dim"]
    # df["n_sample"] = df["n_sample"].astype(int)
    # df["dimension"].fillna(2, inplace=True)
    # df["dimension"] = df["dimension"].astype(int)

    # dftmp = df.copy()
    # dftmp["method"] = dftmp["selection_type"]
    # dftmp["dataset"] = dftmp["dataset_old"]
    # format_table_selrate2achieve_relperf(
    #     table_selrate2achieve_relperf(dftmp),
    #     save_path=os.path.join(result_dir, "table_selrate2achieve_relperf.xlsx"),
    # )

    dftmp = df[
        df["selection_type"].isin(("near_random", "near_filter"))
        & (df["sigma"] == 0.8)
        & (df["dimension"] == 2)
    ].copy()
    dftmp["method"] = dftmp["selection_type"]
    dftmp["method"].fillna("noselect", inplace=True)
    draw_line_perf(
        dftmp,
        methods=dftmp["method"].unique(),
        metric="Accuracy",
        save_dir=os.path.join(result_dir, "final"),
        pdf=True,
        title=False,
        fontsize=14,
    )

    # table_perf(
    #     dftmp,
    #     methods=dftmp["method"].unique(),
    #     metric="Accuracy",
    #     save_dir=os.path.join(
    #         result_dir, "final"
    #     ),
    # )

    for sigma, df_sigma in df.groupby("sigma"):
        for dimension, df_sigma_dim in df_sigma.groupby("dimension"):
            df_sigma_dim = df_sigma_dim.copy()
            df_sigma_dim["method"] = df_sigma_dim["selection_type"]
            df_sigma_dim["method"].fillna("noselect", inplace=True)
            draw_line_perf(
                df_sigma_dim,
                methods=df_sigma_dim["method"].unique(),
                save_dir=os.path.join(
                    result_dir, "line_perf", f"{sigma:.2f}-{dimension}"
                ),
            )

            table_perf(
                df_sigma_dim,
                methods=df_sigma_dim["method"].unique(),
                metric="F1",
                save_dir=os.path.join(
                    result_dir, "table_perf", f"{sigma:.2f}-{dimension}"
                ),
            )

    ##### Natural Datasets
    # there are no improve of selection_type
    # maybe due to the accuracy of estimation of pdoc using the KNN
    # result_dir = r"../output/result/pdoc_innat"
    # df = read_result(result_dir, all_csv=True, duplicated_subset=["exp_id"])
    # df = preprocess_result(
    #     df,
    #     dss=None,
    #     models=None,
    #     methods=None,
    #     d_col_pattern={
    #         "method": {"method": (2, str), "selection_type": (2, str), "selection_rate": (1, float)},
    #         "dataset": {"dataset": (2, str)},
    #     },
    #     metrics2rel=["F1"],
    # )

    # dftmp = df[df["method"]=="pdoc_knn"].copy()
    # dftmp["method"] = dftmp["selection_type"]
    # dftmp["method"].fillna("noselect", inplace=True)
    # table_perf(dftmp,methods=dftmp["method"].unique() , metric="F1", save_dir=os.path.join(result_dir, "table_perf", "F1"))


def analyze_all_in_nat():
    result_dir = RESULT_DIR_NAT
    dss = None
    methods = METHODS
    models = None
    df = read_result(
        result_dir,
        all_csv=True,
        duplicated_subset=["exp_id"],
    )
    df = preprocess_result(
        df,
        d_col_pattern={
            "method": {"method": (2, str), "selection_rate": (1, str)},
            "dataset": {"dataset": (2, str)},
            "model": {"model": (1, str)},
        },
        dss=dss,
        methods=methods,
        models=models,
        metrics2rel=["F1"],
    )
    df = df[(df["selection_rate"] <= 0.5) | (df["method"] == "noselect")]

    df_time = df[df["method"] != "noselect"]
    draw_time(
        df_time,
        save_path=os.path.join(result_dir, "line_perf", "time-samples-nat.png"),
    )

    draw_line_perf(
        df, metric="F1", save_dir=os.path.join(result_dir, "line_perf", "F1")
    )

    table_perf(df, metric="F1", save_dir=os.path.join(result_dir, "table_perf", "F1"))

    format_table_selrate2achieve_relperf(
        table_selrate2achieve_relperf(df),
        save_path=os.path.join(result_dir, "table_selrate2achieve_relperf.xlsx"),
    )


def draw_time(
    df: pd.DataFrame,
    y="time",
    x="raw_train_size",
    renamex="Number of samples",
    renamey="Time (s)",
    save_path=None,
    fontsize=None,
):
    if fontsize is not None:
        plt.rcParams.update({"font.size": fontsize})

    methods_unique = df["method"].unique()
    methods = [m for m in METHODS if m in methods_unique]
    # d_method_order = {m: i for i, m in enumerate(methods)}
    # df = df.sort_values("method", key=lambda x: x.map(d_method_order), inplace=False)
    fig, ax = plt.subplots()
    ax = sns.lineplot(
        data=df,
        x=x,
        y=y,
        hue="method",
        style="method",
        hue_order=methods,
        style_order=methods,
        markers=True,
        dashes=False,
        ax=ax,
    )
    ax.set_yscale("log")
    ax.set_xlabel(renamex)
    ax.set_ylabel(renamey)
    ax.legend(title="Method")
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    if fontsize is not None:
        plt.rcParams.update({"font.size": 10})


def analyze_all_in_syn():
    result_dir = RESULT_DIR_SYN
    dss = None
    methods = METHODS
    models = None
    df = read_result(
        result_dir,
        all_csv=True,
        duplicated_subset=["exp_id"],
    )
    df = preprocess_result(
        df,
        d_col_pattern={
            "method": {"method": (2, str), "selection_rate": (1, float)},
            "dataset": {
                "dataset": (2, str),
                "n_sample": (1, float),
                "sigma": (1, float),
                "dimension": (1, float),
            },
            "model": {"model": (1, str)},
        },
        dss=dss,
        methods=methods,
        models=models,
        metrics2rel=["F1"],
    )
    df = df[(df["selection_rate"] <= 0.5) | (df["method"] == "noselect")]
    df["n_sample"] = df["raw_train_size"] + df["test_size"]
    df["n_sample"] = df["n_sample"].astype(int)
    df["dimension"] = df["train_dim"]
    df["dimension"] = df["dimension"].astype(int)
    df["sigma"].fillna(-1, inplace=True)

    total_num = (
        df["n_sample"].nunique() * df["sigma"].nunique() * df["dimension"].nunique()
    )
    progress_bar = tqdm(total=total_num, desc="analyze_all_in_syn")
    for n_sample, df_sample in df.groupby("n_sample"):
        for sigma, df_sample_sigma in df_sample.groupby("sigma"):
            for dimension, df_sample_sigma_dim in df_sample_sigma.groupby("dimension"):
                draw_line_perf(
                    df_sample_sigma_dim,
                    metric="F1",
                    save_dir=os.path.join(
                        result_dir,
                        "line_perf",
                        "F1",
                        f"{n_sample}-{sigma:.2f}-{dimension}",
                    ),
                )
                table_perf(
                    df_sample_sigma_dim,
                    metric="F1",
                    save_dir=os.path.join(
                        result_dir,
                        "table_perf",
                        "F1",
                        f"{n_sample}-{sigma:.2f}-{dimension}",
                    ),
                )
                progress_bar.update(1)

                format_table_selrate2achieve_relperf(
                    table_selrate2achieve_relperf(df_sample_sigma_dim),
                    save_path=os.path.join(
                        result_dir,
                        "table_selrate2achieve_relperf",
                        f"{n_sample}-{sigma:.2f}-{dimension}.xlsx",
                    ),
                )


def output_syn():
    result_dir = RESULT_DIR_SYN
    fontsize = 14
    output_dir = os.path.join(result_dir, "final")
    dss = (
        "Blobs",
        "Moons",
        "XOR",
        "Circles",
        "Banana",
        "TwoNorm",
    )
    methods = METHODS
    models = ("LR", "SVM", "MLP", "RF", "GBDT", "KNN")
    df = read_result(
        result_dir,
        all_csv=True,
        duplicated_subset=["exp_id"],
    )
    df = preprocess_result(
        df,
        d_col_pattern={
            "method": {"method": (2, str), "selection_rate": (1, float)},
            "dataset": {
                "dataset": (2, str),
                "n_sample": (1, float),
                "sigma": (1, float),
                "dimension": (1, float),
            },
            "model": {"model": (1, str)},
        },
        dss=dss,
        methods=methods,
        models=models,
        metrics2rel=["F1"],
    )
    df = df[(df["selection_rate"] <= 0.5) | (df["method"] == "noselect")]
    df["n_sample"] = df["raw_train_size"] + df["test_size"]
    df["n_sample"] = df["n_sample"].astype(int)
    df["dimension"] = df["train_dim"]
    df["dimension"] = df["dimension"].astype(int)
    df["sigma"].fillna(-1, inplace=True)

    # draw time
    selrate = 0.1
    df_time1 = df[
        (df["method"] != "noselect")
        & (df["dataset"] == "Blobs")
        & (df["sigma"] == 0.8)
        & (df["dimension"] == 2)
        & (df["selection_rate"] == selrate)
    ]
    draw_time(
        df_time1,
        save_path=os.path.join(output_dir, "blobs-08-01-time-samples.pdf"),
        fontsize=fontsize,
    )
    df_time2 = df[
        (df["method"] != "noselect")
        & (df["dataset"] == "Blobs")
        & (df["n_sample"] == 5000)
        & (df["sigma"] == 0.8)
        & (df["selection_rate"] == selrate)
    ]
    draw_time(
        df_time2,
        x="dimension",
        renamex="Dimension",
        save_path=os.path.join(output_dir, "blobs-08-01-time-dimension.pdf"),
        fontsize=fontsize,
    )

    df = df[
        (
            (df["n_sample"] == 5000)
            & (df["dimension"] == 2)
            & (df["sigma"].isin([0.2, 0.8, 1.5]))
        )
        | (df["dataset"].isin(("Banana", "TwoNorm")))
    ]
    format_table_baseline(
        table_baseline(
            df,
            metric="F1",
            models=models,
            index=("dataset", "sigma"),
        ).loc[dss, :, :],
        save_path=os.path.join(output_dir, "baseline.xlsx"),
    )
    df["dataset"] = df["dataset"] + "_" + df["sigma"].astype(str)
    draw_line_perf(
        df,
        metric="F1",
        save_dir=os.path.join(output_dir, "line_perf"),
        title=False,
        pdf=True,
        fontsize=fontsize,
    )
    table_perf(
        df,
        metric="F1",
        save_dir=os.path.join(output_dir, "table_perf"),
    )

    df_pivot = table_selrate2achieve_relperf(df)
    df_pivot = df_pivot.loc[
        ((0.95, 0.98, 0.99), models), [m for m in methods if m != "noselect"]
    ]
    format_table_selrate2achieve_relperf(
        df_pivot,
        save_path=os.path.join(
            output_dir,
            "table_selrate2achieve_relperf.xlsx",
        ),
    )


def output_nat():
    result_dir = RESULT_DIR_NAT
    fontsize = 14
    output_dir = os.path.join(result_dir, "final")
    dss = (
        "Mammographic",
        "Banknote",
        "Obesity",
        "SMS",
        "Mushroom",
        "Nursery",
    )

    methods = METHODS
    models = ("LR", "SVM", "MLP", "RF", "GBDT", "KNN")
    df = read_result(
        result_dir,
        all_csv=True,
        duplicated_subset=["exp_id"],
    )
    df = preprocess_result(
        df,
        d_col_pattern={
            "method": {"method": (2, str), "selection_rate": (1, str)},
            "dataset": {"dataset": (2, str)},
            "model": {"model": (1, str)},
        },
        dss=dss,
        methods=methods,
        models=models,
        metrics2rel=["F1"],
    )
    df = df[(df["selection_rate"] <= 0.5) | (df["method"] == "noselect")]

    df_time = df[df["method"] != "noselect"]
    draw_time(
        df_time,
        save_path=os.path.join(output_dir, "time-samples-nat.pdf"),
        fontsize=fontsize,
    )

    draw_line_perf(
        df,
        metric="F1",
        save_dir=os.path.join(output_dir, "line_perf"),
        title=False,
        pdf=True,
        fontsize=fontsize,
    )

    table_perf(df, metric="F1", save_dir=os.path.join(output_dir, "table_perf"))

    df_pivot = table_selrate2achieve_relperf(df)
    df_pivot = df_pivot.loc[
        ((0.95, 0.98, 0.99), models), [m for m in methods if m != "noselect"]
    ]
    format_table_selrate2achieve_relperf(
        df_pivot,
        save_path=os.path.join(output_dir, "table_selrate2achieve_relperf.xlsx"),
    )

    format_table_baseline(
        table_baseline(
            df,
            metric="F1",
            models=models,
            index="dataset",
        ).loc[dss, :],
        save_path=os.path.join(output_dir, "baseline.xlsx"),
    )


def table_baseline(
    df: pd.DataFrame,
    metric: str = "F1",
    save_path: str = None,
    index="dataset",
    models=MODELS,
):

    df = df[df["method"] == "noselect"]
    df_mean = df.pivot_table(
        values=metric,
        index=index,
        columns="model",
        aggfunc="mean",
    )
    df_std = df.pivot_table(
        values=metric,
        index=index,
        columns="model",
        aggfunc="std",
    )

    df_output = (
        df_mean.map(lambda x: f"{x:.3f}") + "±" + df_std.map(lambda x: f"{x:.3f}")
    )

    models = [m for m in MODELS if m in df_mean.columns]
    df_output = df_output[models]

    if save_path is not None:
        ext = os.path.splitext(save_path)[1]
        if ext == ".csv":
            df_output.to_csv(save_path)
        elif ext == ".xlsx":
            df_output.to_excel(save_path)
        else:
            raise ValueError("The save_path should be a csv or xlsx file.")

    return df_output


def format_table_baseline(df: pd.DataFrame, save_path: str = None):
    """Format the table of baseline performance."""

    def func_row_formatcol(row: pd.Series):
        ser_model_mean = row.map(
            lambda x: float(x.split("±")[0]) if isinstance(x, str) else x
        )

        bestmodelperf = ser_model_mean.max()

        all_bestmodel4perf = ser_model_mean[ser_model_mean == bestmodelperf].index
        ser_method_style = pd.Series("", index=row.index)

        # bold the best model for performance
        ser_method_style[all_bestmodel4perf] = "font-weight: bold"

        return ser_method_style

    df_style = df.style.apply(func_row_formatcol, axis=1)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_style.to_excel(save_path, engine="openpyxl")
    return df_style


def main():
    """Main function."""
    output_syn()
    output_nat()

    # analyze_ours_param()
    # print("`analyze_ours_param` done")

    analyze_all_in_nat()
    print("`analyze_all_in_nat` done")

    analyze_all_in_syn()
    print("`analyze_all_in_syn` done")


if __name__ == "__main__":
    main()
