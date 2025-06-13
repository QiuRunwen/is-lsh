"""
Datasets and data loaders.
"""

import os
import re
import warnings
from pathlib import Path
from typing import Callable, Generator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from tqdm import tqdm

import data
import utils
from ismethod import pdoc
from ismethod.utils import format_y_binary
from preprocess import split_impute_encode_scale

sns.reset_defaults()

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent
DATA_DIR = os.path.join(root_dir, "data")

LOARDERS = (
    data.gen_blobs,
    data.gen_circles,
    data.gen_moons,
    data.gen_xor,
    data.gen_normal_precise_1d,
    data.load_mushroom,
    data.load_mammographic,
    data.load_nursery,
    data.load_obesity,
    data.load_banknote,
    data.load_smsspam,
    data.load_banana,
    data.load_twonorm,
)


class Dataset:
    """数据集类, 用于存储数据集的信息, 以及载入数据集.

    Attributes:
        func_name: 数据集的生成时所用的函数名字.
        kwargs: 数据集的生成函数的参数.
        df: 数据集的DataFrame.
        y_col: 数据集的y列名字.
        name: 数据集的名字.
    """

    __slots__ = ("func_name", "kwargs", "name", "df", "y_col", "_pdoc", "_X", "_y")

    def __init__(
        self,
        func_name: str = None,
        kwargs: dict[str, any] = None,
        df: pd.DataFrame = None,
        y_col: str = None,
        name: str = None,
    ):
        self.func_name = func_name if func_name is not None else ""
        self.kwargs = kwargs if kwargs is not None else {}
        self.df = df if df is not None else pd.DataFrame()
        self.df[y_col] = format_y_binary(self.df[y_col])
        self.y_col = y_col if y_col is not None else ""
        self.name = name if name is not None else ""

        self._pdoc = None
        self._X = None
        self._y = None

    def get_Xy(self) -> tuple[np.ndarray, np.ndarray]:
        if self._X is None:
            self._X = self.df[self.df.columns[self.df.columns != self.y_col]].to_numpy()
        if self._y is None:
            self._y = self.df[self.y_col].to_numpy()
        return self._X, self._y

    def get_pdoc(self, **kwargs_pdoc) -> np.ndarray:
        if self._pdoc is None:
            if self.func_name == "gen_blobs":  # only for blobs Now
                if not kwargs_pdoc:
                    dimension = self.kwargs["dimension"]
                    sigma = self.kwargs["sigma"]
                    kwargs_pdoc = {
                        "neg_mean": [-1] + [0] * (dimension - 1),
                        "neg_cov": sigma[0],
                        "pos_mean": [1] + [0] * (dimension - 1),
                        "pos_cov": sigma[1],
                    }
                X, y = self.get_Xy()
                self._pdoc = pdoc.calc_pdoc_gaussian(
                    X=X,
                    y=y,
                    **kwargs_pdoc,
                )
            else:
                raise NotImplementedError
        return self._pdoc

    def gen_train_test(
        self,
        test_size: float = 0.2,
        random_state: int = None,
        return_array: bool = False,
    ):
        return split_impute_encode_scale(
            df=self.df,
            y_col=self.y_col,
            test_size=test_size,
            seed=random_state,
            return_array=return_array,
        )

    def __repr__(self) -> str:
        return f"name: {self.name}, df.shape: {self.df.shape}"


# `synthetic.gen_power_grid.__kwdefaults__` may miss the args which don't have default value
# `inspect.signature` can solve this problem
def _get_dict_loader() -> (
    dict[str, tuple[Callable[..., tuple[pd.DataFrame, str]], dict[str, any]]]
):
    """Get all loaders with names. return `{dataset_name: (loader, default_paramters)}`"""
    dict_name_loader = {
        loader.__name__: (loader, utils.get_function_default_params(loader)[0])
        for loader in LOARDERS
    }
    for k, v in dict_name_loader.items():
        if "data_dir" in v[1]:
            v[1]["data_dir"] = DATA_DIR

    return dict_name_loader


def get_funcname_kwargs(func_names: list = None) -> dict[str, dict]:
    """Get loader and kwargs. return `{loader_name: kwargs}`"""
    func_names = (
        func_names if func_names is not None else list(_get_dict_loader().keys())
    )
    default_dict_loader = _get_dict_loader()
    dict_funcname_kwargs = {
        func_name: default_dict_loader[func_name][1] for func_name in func_names
    }
    return dict_funcname_kwargs


def load_one(func_name: str, kwargs: dict = None) -> Dataset:
    """Load a dataset. use `get_funcname_kwargs` to get all func_name and default kwargs.

    Args:
        func_name (str): The loader name of the dataset. Defaults to None.
        kwargs (dict, optional): The kwargs of the loader.
            It will use default kwargs if it is None. Defaults to None.

    Returns:
        Dataset: The dataset.
    """
    dict_name_loader = _get_dict_loader()
    load_func, default_kwargs = dict_name_loader[func_name]
    kwargs = kwargs if kwargs is not None and len(kwargs) != 0 else default_kwargs
    df, y_col = load_func(**kwargs)
    return Dataset(
        func_name=func_name, kwargs=kwargs, df=df, y_col=y_col, name=func_name
    )


def load_all(
    dict_funcname_kwargs: dict[str, dict] = None,
) -> Generator[Dataset, None, None]:
    """Load datasets. Defaut load all.
    use `get_funcname_kwargs` to get all func_name and default kwargs.

    Args:
        dict_funcname_kwargs (dict[str, dict], optional): `{func_name: kwargs}`. Defaults to None.

    Yields:
        Generator[Dataset, None, None]: 载入的数据集.
    """
    dict_funcname_kwargs = (
        get_funcname_kwargs() if dict_funcname_kwargs is None else dict_funcname_kwargs
    )

    for func_name, kwargs in dict_funcname_kwargs.items():
        yield load_one(func_name, kwargs)


def load_one_assign_name(
    func_name: str, kwargs: dict = None, name: str = None
) -> Dataset:
    """`load_one`的拓展, 进一步给数据集命名.
    use `get_funcname_kwargs` to get all func_name and default kwargs.

    Args:
        func_name (str): 数据集载入的函数名.
        kwargs (dict, optional): 数据集载入用到的参数. Defaults to None.
        name (str, optional): 数据集的名字. Defaults to None.

    Returns:
        Dataset: _description_
    """
    dataset = load_one(func_name, kwargs)
    dataset.name = name if name is not None else func_name
    return dataset


def load_all_assign_name(
    dict_datasetname_funcname_kwargs: dict[str, dict[str, any]] = None
) -> Generator[Dataset, None, None]:
    """`load_all`的拓展, 进一步给数据集命名. 默认载入所有数据集. kwargs为None时使用默认参数.
    use `get_funcname_kwargs` to get all func_name and default kwargs.

    Args:
        dict_datasetname_funcname_kwargs (dict[str, dict[str, any]], optional):
            `{dataset_name:{func_name:str, kwargs:dict}}`. Defaults to None.

    Yields:
        Generator[Dataset, None, None]: _description_
    """

    if dict_datasetname_funcname_kwargs is None:
        dict_funcname_kwargs = get_funcname_kwargs()
        dict_datasetname_funcname_kwargs = {}
        for func_name, kwargs in dict_funcname_kwargs.items():
            dataset_name = "_".join(
                func_name.split("_")[1:]
            )  # "gen_power_grid" -> "power_grid"
            dict_datasetname_funcname_kwargs[dataset_name] = {
                "func_name": func_name,
                "kwargs": kwargs,
            }

    for dataset_name, dict_tmp in dict_datasetname_funcname_kwargs.items():
        func_name = dict_tmp["func_name"]
        kwargs = dict_tmp.get("kwargs", None)
        yield load_one_assign_name(func_name, kwargs, dataset_name)


def calc_col_card(df: pd.DataFrame, y_col: str = None):
    """Calculate the cardinality of each column in a dataset.
    For categorical feature, the cardinality is its number of unique categories (NAN also count).
    For numerical feature, the cardinality is np.nan for compatibility and caculation of effective cardinality of a dataset.

    Args:
        df (pd.DataFrame): A dataset with correct dtype, especially the `category` columns
        y_col (str, optional): _description_. Defaults to None.

    Returns:
        pd.Series: ser_col_card with descending cardinality.
    """
    X = df if y_col is None else df[[col for col in df.columns if col != y_col]]
    cols_cat = X.columns[X.dtypes == "category"]
    # cols_num = [col for col in X.columns if col not in cols_cat]
    ser_col_card = X.apply(
        lambda ser: len(ser.unique()) if ser.name in cols_cat else np.nan
    )

    return ser_col_card.sort_values(ascending=False)


def calc_col_importance(
    X: pd.DataFrame,
    y: pd.Series,
    name: str = "",
    seed=1,
    save_dir="./",
    use_cache=True,
):
    prefix = os.path.join(save_dir, name + "_imp" + "_s=" + str(seed))
    filename = prefix + ".pkl"
    if use_cache and os.path.exists(filename):
        ser_col_importance = utils.load(filename)
    else:
        X = X.copy(deep=True)

        # 1. numerical variable
        #        replace inf by max+1
        #        replace -inf by min-1
        #        filling na by mean (excluding na)
        # 2. encode categorical variable by TargetEncoding
        cat_cols = []
        for col in X:
            if pd.api.types.is_numeric_dtype(X[col]):
                col_max = X[col].max()
                if col_max == np.inf:
                    msg = f"Data set: {name} col: {col} np.inf -> max+1"
                    warnings.warn(msg)
                    new_col = X[col].replace([np.inf], np.nan)
                    col_max = new_col.max()
                    X[col].replace([np.inf], col_max + 1, inplace=True)
                col_min = X[col].min()
                if col_min == -np.inf:
                    msg = f"Data set: {name} col: {col} -np.inf -> min-1"
                    warnings.warn(msg)
                    new_col = X[col].replace([-np.inf], np.nan)
                    col_min = new_col.min()
                    X[col].replace([-np.inf], col_min - 1, inplace=True)

                v = X[col].mean()
                X[col] = X[col].fillna(v)
            elif isinstance(X[col].dtype, pd.CategoricalDtype):
                cat_cols.append(col)
            else:
                warnings.warn(
                    f"col: {col} is not numeric or categorical. it will be ignored."
                )

        y = y.astype("int") if y.nunique() == 2 else y.astype("float")
        X = TargetEncoder(cols=cat_cols).fit_transform(X, y)

        model = (
            RandomForestClassifier(
                random_state=seed,
                n_estimators=100,  # default 100
                criterion="gini",  # default 'gini'
                max_depth=30,  # default None, no limit
                min_samples_split=20,  # default 2
                min_samples_leaf=1,  # default 1
            )
            if y.nunique() == 2
            else RandomForestRegressor(
                random_state=seed,
                n_estimators=100,
                criterion="squared_error",
                max_depth=30,
                min_samples_split=20,
                min_samples_leaf=1,
            )
        )
        model.fit(X, y)

        ser_col_importance = pd.Series(model.feature_importances_, index=X.columns)
        utils.save(ser_col_importance, filename)

    # plot figure
    ser_col_importance.sort_values(ascending=True, inplace=True)
    fig, ax = plt.subplots(figsize=(20, len(ser_col_importance) / 2))
    ax.grid(True)
    ser_col_importance.plot(kind="barh", ax=ax)
    ax.set_xlabel("variable importance")
    plt.savefig(prefix + ".png")
    plt.close(fig)
    return ser_col_importance


def get_num_cat_cols(df: pd.DataFrame, y_col: str = None):
    """num_cols, cat_cols, cat_is_num_cols, cat_miss_lable_cols"""
    num_cols = []
    cat_cols = []
    cat_is_num_cols = []
    cat_miss_lable_cols = []
    for col in df.columns:
        if col == y_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            num_cols.append(col)
        elif isinstance(df[col].dtype, pd.CategoricalDtype):
            cat_cols.append(col)
            if df[col].astype(str).str.isdigit().all():
                cat_is_num_cols.append(col)
        else:
            cat_miss_lable_cols.append(col)
            cat_cols.append(col)

    return num_cols, cat_cols, cat_is_num_cols, cat_miss_lable_cols


def draw_all_cat_hist(df: pd.DataFrame, y_col: str, output_dir: str, overwrite=False):
    _, cat_cols, _, _ = get_num_cat_cols(df, y_col)
    for col in cat_cols:
        # Define a regular expression pattern to match illegal characters
        illegal_chars_pattern = r'[\\/:\*\?"<>\|]'
        # Replace illegal characters with underscores
        tmp_col = re.sub(illegal_chars_pattern, "_", col)
        fp = os.path.join(output_dir, f"cat_col={tmp_col}.png")
        if os.path.exists(fp) and not overwrite:
            continue
        fig, ax = plt.subplots()
        ser_val_count = df[col].value_counts(sort=True, dropna=False)
        # ser_val_count.plot(kind="barh", ax=ax)
        sns.lineplot(x=np.arange(ser_val_count.size), y=ser_val_count.values, ax=ax)
        fig.suptitle(col)
        os.makedirs(output_dir, exist_ok=True)

        fig.savefig(fp)
        plt.close()


def get_data_desc(
    df: pd.DataFrame,
    y_col: str = None,
    high_card_col_imp: bool = False,
    imp_output_dir: str = None,
    use_cache: bool = True,
):
    X = df if y_col is None else df[[col for col in df.columns if col != y_col]]
    cat_cols = X.columns[X.dtypes == "category"]

    major_class_ratio = None
    pos_class_ratio = None
    isTaskReg = None
    if y_col is not None:
        ser_class_count = df[y_col].value_counts()
        isTaskReg = ser_class_count.shape[0] != 2
        if ser_class_count.shape[0] == 2:  # only support binaray classification
            major_class_ratio = (ser_class_count / df.shape[0]).iloc[0]
            pos_class_ratio = (ser_class_count / df.shape[0]).loc[1]

    ser_col_card = calc_col_card(X)  # only calculate the cardinality of X
    res_dict = {
        "row": X.shape[0],
        "num_col": X.shape[1] - cat_cols.size,
        "cat_col": cat_cols.size,
        "max_card": ser_col_card.iloc[0],
        "sum_card": ser_col_card.fillna(1).sum(),
        "major_class_ratio": major_class_ratio,
        "pos_class_ratio": pos_class_ratio,
        "isTaskReg": isTaskReg,
    }
    df_col_card_importance = ser_col_card.to_frame("cardinality")
    if high_card_col_imp:
        y = df[y_col]
        ser_col_importance = calc_col_importance(
            X, y, save_dir=imp_output_dir, use_cache=use_cache
        )
        df_col_card_importance["importance"] = ser_col_importance
        max_card_col = ser_col_card.index[0]
        res_dict["max_card_col"] = max_card_col
        res_dict["imp_of_max_card_col"] = ser_col_importance.loc[max_card_col]
        res_dict["sum_cat_importance"] = ser_col_importance[cat_cols].sum()

    if imp_output_dir is not None:
        os.makedirs(imp_output_dir, exist_ok=True)
        df_col_card_importance.to_csv(
            os.path.join(imp_output_dir, "col_card_importance.csv")
        )

    return res_dict


def draw_scatter(
    dss: list[Dataset],
    pdoc_calc_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
    draw_in_one: bool = False,
    savr_dir: str = None,
):
    """绘制散点图."""

    if draw_in_one:
        nrows = 3
        ncols, remainder = divmod(len(dss), nrows)
        if remainder != 0:
            ncols += 1
        ncols = max(ncols, 2)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))

    for i, ds in enumerate(dss):
        df = ds.df.copy()
        y_col = ds.y_col
        df[y_col] = df[y_col].astype("category")
        X_cols = df.columns[df.columns != y_col]

        if draw_in_one:
            idx_j, idx_i = divmod(i, nrows)
            ax = axes[idx_i][idx_j]
        else:
            fig, ax = plt.subplots()

        if df.shape[1] == 2:
            # y, x1
            plot_hue_col = y_col
            if pdoc_calc_func is not None:
                X = df[X_cols]
                y = df[y_col] if y_col is not None else None
                df[pdoc_calc_func.__name__] = pdoc_calc_func(X, y)
                plot_hue_col = pdoc_calc_func.__name__
            sns.histplot(
                data=df,
                x=X_cols[0],
                hue=plot_hue_col,
                ax=ax,
            )
        elif df.shape[1] == 3:
            # y, x1, x2
            plot_hue_col = y_col
            plot_style_col = None
            if pdoc_calc_func is not None:
                X = df[X_cols]
                y = df[y_col] if y_col is not None else None
                df[pdoc_calc_func.__name__] = pdoc_calc_func(X, y)
                plot_hue_col = pdoc_calc_func.__name__
                plot_style_col = y_col
            sns.scatterplot(
                data=df,
                x=X_cols[0],
                y=X_cols[1],
                hue=plot_hue_col,
                style=plot_style_col,
                ax=ax,
            )
        else:
            print(f"`draw_scatter` Skip {ds.name} for its shape is {df.shape}")
            continue

        if draw_in_one:
            ax.set_title(ds.name)
        else:
            if savr_dir is not None:
                os.makedirs(savr_dir, exist_ok=True)
                fig.savefig(os.path.join(savr_dir, f"{ds.name}.png"))
                plt.close()
            else:
                plt.show()

    if draw_in_one:
        plt.show()


def desc_alldataset(
    data_dir=DATA_DIR,
    output_dir="../output/data_desc",
    draw_reg=True,
    high_card_col_imp=False,
    use_cache=True,
):
    res_ls = []
    os.makedirs(output_dir, exist_ok=True)
    progress_bar = tqdm(total=len(LOARDERS))
    for load_func in LOARDERS:
        name = load_func.__name__
        progress_bar.set_description(name)
        is_data_syn = name.startswith("gen_")
        if is_data_syn:
            df, y_col = load_func()
        else:
            df, y_col = load_func(data_dir=data_dir)
        output_dir_dataset = os.path.join(output_dir, name)

        if df.shape[1] in (2, 3):
            draw_scatter(
                [Dataset(func_name=name, df=df, y_col=y_col, name=name)],
                draw_in_one=False,
                savr_dir=output_dir_dataset,
            )
        output_dir_dataset_valcounts = os.path.join(output_dir_dataset, "valcounts")
        draw_all_cat_hist(df, y_col, output_dir_dataset_valcounts)
        res = get_data_desc(df, y_col, high_card_col_imp, output_dir_dataset, use_cache)
        res["name"] = name
        res_ls.append(res)
        if draw_reg:
            if res["isTaskReg"]:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                sns.lineplot(
                    x=np.arange(df.shape[0]),
                    y=df[y_col].sort_values().values,
                    ax=axes[0],
                )
                sns.histplot(data=df, x=y_col, ax=axes[1])
                fig.suptitle(name)
                fp = os.path.join(output_dir, f"{name}.png")
                fig.savefig(fp)
                plt.close()
                print(
                    f"Target in regression has been drawn and saved in `{os.path.abspath(fp)}`"
                )
        progress_bar.update(1)

    df_res = pd.DataFrame(res_ls)
    df_res["sample_per_level"] = df_res["row"] / df_res["max_card"]

    # Reorder the columns with 'name' in the first position
    df_res = df_res[["name"] + [col for col in df_res.columns if col != "name"]]

    df_res.sort_values("row", inplace=True)
    fp = os.path.join(output_dir, "alldata_desc.csv")
    df_res.to_csv(fp, index=False)
    print(f"The description of all dataset has been saved in `{os.path.abspath(fp)}`")
    print(df_res)
    return df_res


def draw4thesis(save_dir=None, fontsize=None):

    names = ["blobs", "circles", "moons", "xor"]
    sigmas = [0.2, 0.8, 1.5]
    dict_dataset_funcname_kwargs = {}
    for name in names:
        for sigma in sigmas:
            dict_dataset_funcname_kwargs[f"{name}_{sigma}"] = {
                "func_name": f"gen_{name}",
                "kwargs": {"n_samples": 5000, "sigma": sigma, "seed": 42},
            }
    dict_dataset_funcname_kwargs["circles_0_24"] = {
        "func_name": "gen_circles",
        "kwargs": {"n_samples": 24, "sigma": 0, "seed": 42},
    }

    dict_dataset_funcname_kwargs["moons_0_10"] = {
        "func_name": "gen_moons",
        "kwargs": {"n_samples": 10, "sigma": 0, "seed": 42},
    }

    dict_dataset_funcname_kwargs["banana"] = {
        "func_name": "load_banana",
        "kwargs": {},
    }

    if fontsize is not None:
        plt.rcParams["font.size"] = fontsize
    fig, ax = plt.subplots()
    for ds in load_all_assign_name(dict_dataset_funcname_kwargs):
        df, y_col = ds.df.copy(), ds.y_col
        x1col, x2col = df.columns[df.columns != y_col]
        df[y_col] = df[y_col].astype("category")
        ax = sns.scatterplot(data=df, x=x1col, y=x2col, hue=y_col, style=y_col, ax=ax)
        if ds.name in ("circles_0_24", "moons_0_10"):
            ax.grid(True)
            ax.set_aspect("equal", adjustable="box")
        else:
            # reset
            ax.grid(False)
            ax.set_aspect("auto", adjustable="box")
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(
                os.path.join(save_dir, f"ds_{ds.name}.png"), bbox_inches="tight"
            )
            ax.cla()
        else:
            plt.show()
            fig, ax = plt.subplots()

    plt.close(fig)
    if fontsize is not None:
        plt.rcParams["font.size"] = 10


if __name__ == "__main__":
    draw4thesis(save_dir="../paper/latex/assets", fontsize=14)
    print("`draw4thesis` done")

    desc_alldataset()
    print("`desc_alldataset` done")
