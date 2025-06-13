import logging

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from category_encoders import TargetEncoder

from ismethod.utils import format_y_binary


def get_num_cat_cols(df: pd.DataFrame, y_col: str):
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
                logging.warning("dataset:%s, col:%s, 是数字型的分类变量", df.shape, col)
                cat_is_num_cols.append(col)
        else:
            cat_miss_lable_cols.append(col)
            logging.warning("dataset:%s, col:%s, dtype不是数字也不是category", df.shape, col)
            cat_cols.append(col)

    return num_cols, cat_cols, cat_is_num_cols, cat_miss_lable_cols


def _format_df(df: pd.DataFrame, y_col: str):
    X = df.drop(columns=[y_col])
    y = df[y_col].copy()
    num_cols, cat_cols, cat_is_num_cols, cat_miss_lable_cols = get_num_cat_cols(
        df, y_col
    )
    # TODO category_encoders的函数默认只对object的array转换(有指明category的df也行)，
    # 而在imputer返回的是原数据类型的array，如果原先dtype全为数值，则category_encoders失效。因此数值型的category最好转换下
    X[cat_is_num_cols] = X[cat_is_num_cols].astype(str).astype("category")
    X[cat_miss_lable_cols] = X[cat_miss_lable_cols].astype(str).astype("category")
    # X[cat_cols] = X[cat_cols].astype(str).astype("category")

    # y = y.astype("float")

    for col in num_cols:
        col_max = X[col].max()
        if col_max == np.inf:
            logging.warning("Data set: %s col: %s np.inf -> max+1", df.shape, col)
            new_col = X[col].replace([np.inf], np.nan)
            col_max = new_col.max()
            X[col].replace([np.inf], col_max + 1, inplace=True)

        col_min = X[col].min()
        if col_min == -np.inf:
            logging.warning("Data set: %s col: %s -np.inf -> min-1", df.shape, col)
            new_col = X[col].replace([-np.inf], np.nan)
            col_min = new_col.min()
            X[col].replace([-np.inf], col_min - 1, inplace=True)
    return X, y, num_cols, cat_cols


def split_impute_encode_scale(
    df: pd.DataFrame,
    y_col: str,
    test_size: float,
    seed: int = None,
    return_array: bool = True,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
):
    df_X, ser_y, num_cols, cat_cols = _format_df(df, y_col)

    # since we use TargetEncoder, when target is binary, we need to convert it to 0, 1
    ser_y_unique = ser_y.unique()
    if len(ser_y_unique) == 2:
        if 0 not in ser_y_unique:
            ser_y[:] = format_y_binary(ser_y.to_numpy(), neg_and_pos=False)

    X_train_df, X_test_df, y_train_ser, y_test_ser = train_test_split(
        df_X, ser_y, test_size=test_size, random_state=seed
    )
    cat_pipeline = Pipeline(
        steps=[
            # ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", TargetEncoder()),
            ("scaler", StandardScaler()),  # 很多涉及到距离算法，所以标准化会好些
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    trans = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
    ).set_output(
        transform="pandas"
    )  # return pandas.DataFrame instead of numpy.ndarray

    trans.fit(X_train_df, y_train_ser)

    # trans will keep the column names, but the order is not guaranteed
    # the order is the same as the order of the columns
    # in the ColumnTransformer(transformers=[...]), num_cols and cat_cols
    X_train = trans.transform(X_train_df)
    X_test = trans.transform(X_test_df)

    # keep the column names and order
    X_train.columns = X_train.columns.str.replace(r"num__|cat__", "", regex=True)
    X_train = X_train[X_train_df.columns]
    X_test.columns = X_test.columns.str.replace(r"num__|cat__", "", regex=True)
    X_test = X_test[X_test_df.columns]

    y_train = y_train_ser
    y_test = y_test_ser

    if return_array:
        return (
            X_train.to_numpy(),
            X_test.to_numpy(),
            y_train.to_numpy(),
            y_test.to_numpy(),
        )
    return X_train, X_test, y_train, y_test
