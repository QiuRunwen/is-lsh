# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 15:11:28 2021

@author: iwenc, QiuRunwen
"""

import numpy as np
import pandas as pd


def find_useless_colum(
    df: pd.DataFrame,
    max_missing_ratio=0.5,
    min_rows_per_value=2,
    max_ratio_per_cat=1,
    verbose=False,
):
    """
    Identify useless columns from a DataFrame.
    Columns are divided into three types:
        num_float: contains numeric values only and at least a number that is not integer
        num_int:   contains numeric values and all values are integers (3.0 is treated as an integer)
        cat_like:  none numeric values are considered like categorical

    If column is considered useless and classified into the following types:
        empty:              if the column contains no value
        singel-valued:      if the column contains only one value
        id-like:            A num_int or cat_like column contains a unique
                            value for each sample. It is okay for a num_float
                            column to contain a unqiue value for each sample
        too-many-missing:   if a column contains too many missing values --
                            exceeding total number of samples * max_missing_ratio
        too-small-cat:      if average samples per category are too few in a
                            cat_like column -- less than min_rows_per_value
        too-large-cat:      if a single category in a cat-like column contains
                            too many samples -- exceeding total number of
                            samples * max_ratio_per_cat

    Parameters
    ----------
    df : pandas.DataFrame
        A table contains many columns
    max_missing_ratio : float in [0.0,1.0], optional
        Threshold for determining a column to be too-many-missing. The default is 0.5.
    min_rows_per_value : int, optional
        Threshold for determining a column to be too-small-cat. The default is 2.
    max_ratio_per_cat : float in [0.0,1.0], optional
        Threshold for determining a column to be too-large-cat. The default is 0.9.
    verbose : bool, optional
        If True print more messages. The default is False.

    Returns
    -------
    dict
        A dictionary where a key represents a type of useless columns and
        the value is a list of useless columns of the corresponding type.

    """
    empty_cols = []
    single_value_cols = []
    id_like_cols = []
    too_many_missing_cols = []
    too_small_cat_cols = []
    too_large_cat_cols = []

    # TODO: one-to-one map (two columns are one-to-one map), e.g. Crime 'AREA' and 'AREA NAME'
    # TODO: nearly one-to-one map  e.g. 'Premis Cd', 'Premis Desc' (both 803 and 805 are mapped to 'RETIRED (DUPLICATE) DO NOT USE THIS CODE'), rest is one-to-one)
    # TODO: nearly one-to-one map  e.g. Crime 'Weapon Used Cd', 'Weapon Desc', 222 -> np.nan. The rest is one-to-one
    row_count = df.shape[0]
    # Cannot convert non-finite values (NA or inf) to integer
    inf_dropped = df.replace([np.inf, -np.inf], np.nan, inplace=False)
    for col in df:
        missing_count = df[col].isna().sum()
        if missing_count == row_count:
            if verbose:
                print(f"{col=} contains no value.")
            empty_cols.append(col)
            continue

        vc = df[col].value_counts(sort=True, dropna=True)
        if vc.size == 1:
            if missing_count == 0:
                if verbose:
                    print(f"{col=} contains a single value: {vc.index[0]}")
            else:
                if verbose:
                    print(
                        f"{col=} contains a single value and missing value: {vc.index[0]}"
                    )
            single_value_cols.append(col)
            continue

        na_dropped = inf_dropped[col].dropna()
        if not pd.api.types.is_numeric_dtype(na_dropped):
            col_type = "cat_like"
        elif np.array_equal(na_dropped, na_dropped.astype(int)):
            col_type = "num_int"
        else:
            col_type = "num_float"

        # a unique value for each record
        if vc.size == row_count and col_type != "num_float":
            if col_type == "cat_like":
                if verbose:
                    print(f"cat_like column: {col} has unique value for each row")
                id_like_cols.append(col)
                continue
            else:  # col_type == 'num_int'
                print(f"warning: int column: {col} has unique value for each row.")

        # a unique value for each record that has value
        if vc.size + missing_count == row_count and col_type != "num_float":
            if col_type == "cat_like":
                if verbose:
                    print(
                        f"cat_like column: {col} has unique value for each row that has value"
                    )
                id_like_cols.append(col)
                continue
            else:  # col_type == 'num_int'
                if verbose:
                    print(
                        f"warning: int column: {col} has unique value for each row that has value"
                    )

        # missing rate exceed max_missing_ratio
        missing_count = df[col].isna().sum()
        if missing_count > max_missing_ratio * row_count:
            if verbose:
                print(
                    f"{col=} has too many missing values: {missing_count}, missing ratio > {max_missing_ratio=}"
                )
            too_many_missing_cols.append(col)
            continue

        # too few records per category
        if vc.size > 0:
            rows_per_value = row_count / vc.size
        else:
            rows_per_value = 0
        if rows_per_value < min_rows_per_value and col_type != "num_float":
            if col_type == "cat_like":
                if verbose:
                    print(
                        f"cat_like column: {col} rows per cat: {rows_per_value} < {min_rows_per_value=}"
                    )
                too_small_cat_cols.append(col)
                continue
            else:  # col_type == 'num_int':
                if verbose:
                    print(
                        f"warning: int column: {col} rows per cat: {rows_per_value} < {min_rows_per_value=}"
                    )

        max_rows_per_cat = row_count * max_ratio_per_cat
        if vc.size > 0 and vc.iloc[0] > max_rows_per_cat:
            if col_type == "cat_like":
                if verbose:
                    print(
                        f"cat_like column: {col} rows for largest cat {vc.index[0]}: {vc.iloc[0]} > {max_ratio_per_cat=}"
                    )
                too_large_cat_cols.append(col)
                continue
            else:  # col_type == 'num_int':
                if verbose:
                    print(
                        f"warning: int column: {col} rows for largest cat {vc.index[0]}: {vc.iloc[0]} > {max_ratio_per_cat=}"
                    )

    return {
        "empty_cols": empty_cols,
        "single_value_cols": single_value_cols,
        "id_like_cols": id_like_cols,
        "too_many_missing_cols": too_many_missing_cols,
        "too_small_cat_cols": too_small_cat_cols,
        "too_large_cat_cols": too_large_cat_cols,
    }


def drop_useless(
    df: pd.DataFrame, useless_cols_dict: dict[str, list[str]], verbose=True
):
    """
    Drop useless columns identified by find_useless_colum(df) from a dataframe df:
        drop(df, find_useless_colum(df))

    Parameters
    ----------
    df : pandas.DataFrame
        A data table.
    useless_cols_dict : dict(type,list)
        Use less columns identified by find_useless_colum(df,...)
    verbose : bool, optional
        If true print more messages. The default is True.

    Returns
    -------
    df : pandas.DataFrame
        A copy of df with use less columns dropped.

    """
    for useless_type in useless_cols_dict:
        cols = useless_cols_dict[useless_type]
        if verbose:
            print(f"drop {useless_type}: {cols}")
        df = df.drop(columns=cols)
    return df


def is_one_to_one(df: pd.DataFrame, col1: str, col2: str):
    """
    Check if col1 and col2 is one-to-one mapping

    Parameters
    ----------
    df : pandas.DataFrame
        A table
    col1 : string
        Name of a column in df
    col2 : string
        Name of a column in df

    Returns
    -------
    pd.Series
        If col1 and col2 is one-to-one mapping, return a series where index is value in col1 and value is value in col2;
        None otherwise.

    """
    dfu = df.drop_duplicates([col1, col2])
    a = dfu.groupby(col1)[col2].count()
    b = dfu.groupby(col2)[col1].count()
    if a.max() == 1 and a.min() == 1 and b.max() == 1 and b.min() == 1:
        return pd.Series(dfu[col2].values, index=dfu[col1].values)
    return None


def sampling_by_class(
    df: pd.DataFrame,
    class_col: str,
    num_sample: int = None,
    ratio: float = None,
    seed: int = 1234,
):
    """
    Sample by class, reset index and remove unused categories

    Parameters
    ----------
    df : pandas.DataFrame
        data
    class_col : str
        name of class label, where 1: positive class; 0: negative class
    num_sample : int
        number of instances to sample
    ratio : float
        0=< ratio <=1
    seed : int

    Returns
    -------
    df : pandas.DataFrame
        The sampled

    """

    ratio = num_sample / df.shape[0] if ratio is None else ratio

    df = df.groupby(class_col).apply(
        lambda data: data.sample(frac=ratio, replace=False, random_state=seed)
    )

    df = df.reset_index(drop=True).sample(frac=1)

    # df_tmp = df[class_col].to_frame()
    # df_tmp['index'] = np.arange(df_tmp.shape[0])

    # df_tmp = df_tmp.groupby(class_col).apply(lambda data: data.sample(frac=ratio, replace=False, random_state=seed))

    # df = df.iloc[df_tmp['index'],:].reset_index(drop=True)#.sample(frac=1)

    # after sampling some levels in a categorical variable may
    # not have any data, remove `unused categories'
    for col in df:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].cat.remove_unused_categories()

    return df


if __name__ == "__main__":
    a = pd.DataFrame(
        {
            "a": [1, 2, np.nan, 3, 4],
            "b": pd.Series([3, 4, 4, np.nan, 5]).astype("category"),
            "c": [np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )
    # b = fillna(a)
