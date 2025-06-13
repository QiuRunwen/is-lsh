"""
This module contains some utility functions for ismethod.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from sklearn import impute, preprocessing
from sklearn.preprocessing import LabelEncoder


def format_y_binary(y: np.ndarray, neg_and_pos: bool = True) -> np.ndarray:
    """Format binary label to -1 and 1 (or 0 and 1). The raw label 1 will be transformed to 1
    , and the other one will be transformed to -1 (or 0).

    Args:
        y (np.ndarray): The raw labels.
        neg_and_pos (bool, optional): Whether to transform the label to -1 and 1. Defaults to True.
            False means to transform the label to 0 and 1.

    Returns:
        np.ndarray: The transformed labels.
    """

    y_unique = np.unique(y)
    assert y_unique.size == 2

    if not isinstance(y, np.ndarray):
        y = np.array(y)

    if 1 in y_unique:
        if neg_and_pos:
            y_final = np.where(y == 1, 1, -1)
        else:
            y_final = np.where(y == 1, 1, 0)
    else:
        y_final = LabelEncoder().fit_transform(y)
        if neg_and_pos:
            y_final = y_final * 2 - 1

    return y_final


def get_distance_matrix(data: pd.DataFrame | np.ndarray):
    """Get the distance matrix of the data."""

    # TODO optimize the performance by using `pdist`
    if type(data) == pd.DataFrame:
        ser_col_isnumeric: pd.Series = data.dtypes.transform(
            pd.api.types.is_numeric_dtype
        )
        if not ser_col_isnumeric.all():
            ser_col_col = ser_col_isnumeric.index.to_series()
            col_isNumeric = ser_col_col[ser_col_isnumeric].values
            col_nonNumeric = ser_col_col[~ser_col_isnumeric].values
            warnings.warn(
                f"There are some non-numeric columns `{col_nonNumeric}`, which will be ignore."
            )
            data = data[col_isNumeric]

        dm = distance_matrix(data.values, data.values)
    else:
        dm = distance_matrix(data, data)

    return dm


def sort_distance_matrix(dm: np.ndarray):
    """Sort the distance matrix by row. The first column is the nearest point to the first point, and so on."""
    assert dm.shape[0] == dm.shape[1]
    n_point = dm.shape[0]
    # set the diagonal to inf to avoid the first point to be the nearest point to itself
    for i in range(n_point):
        dm[i, i] = np.inf

    dm_sorted = np.sort(dm)  # sort by row ascending
    return dm_sorted


def preprocess_data(data: "pd.DataFrame | np.ndarray", drop_duplicate=False):
    """Preprocess numercial dataset by z-score standardization, median imputation, drop duplicates.

    Args:
        data (pd.DataFrame | np.ndarray): _description_
        drop_duplicate (bool, optinal): whether to drop duplicate rorws in `X`.

    Returns:
        np.ndarray or (np.ndarray, StandardScaler,SimpleImputer): _description_
    """

    standard_scaler = preprocessing.StandardScaler()
    simple_imputer = impute.SimpleImputer(strategy="median")
    data = standard_scaler.fit_transform(simple_imputer.fit_transform(data))

    if drop_duplicate:
        sample_before = data.shape[0]
        data = pd.DataFrame(data).drop_duplicates().values
        sampel_after = data.shape[0]
        sample_drop = sample_before - sampel_after
        if sample_drop != 0:
            print(
                f"Sample raw: {sample_before}. After drop {sample_drop} duplicates. {sampel_after} left"
            )

    return data
