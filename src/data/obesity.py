# -*- coding: utf-8 -*-
"""
Estimation of obesity levels based on eating habits and physical condition
https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition

@author: QiuRunwen
"""

import os
import pandas as pd

if __name__ == "__main__":
    import util  # 单独运行本脚本用
else:
    from . import util  # 当作一个module被别人 import 时用


def load_obesity(
    data_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data"), drop_useless=True, num_sample=None, verbose=False
):
    # 1. read/uncompress data

    file_path = os.path.join(
        data_dir, "obesity", "ObesityDataSet_raw_and_data_sinthetic.csv"
    )
    df = pd.read_csv(file_path)

    # 2111*17

    # 2. convert numeric/categorical columns
    cat_cols = [col for col in df.columns if col not in ["Age", "Height", "Weight"]]
    for col in cat_cols:
        if not isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype("category")

    # 3. simple feature extraction

    # 4. compute class label
    # NObeyesdad
    # Obesity_Type_I         351
    # Obesity_Type_III       324
    # Obesity_Type_II        297
    # Overweight_Level_I     290
    # Overweight_Level_II    290
    # Normal_Weight          287
    # Insufficient_Weight    272
    # Name: count, dtype: int64

    y_col = "NObeyesdad"
    df[y_col] = df[y_col].map(lambda x: 1 if x.startswith("Obesity") else 0)
    assert df[y_col].notna().all()

    # after convert to binary class
    # NObeyesdad
    # 0    1139
    # 1     972

    # 5. drop_useless
    if drop_useless:
        # a. manually remove useless columns

        # b. auto identified useless columns
        useless_cols_dict = util.find_useless_colum(df)
        df = util.drop_useless(df, useless_cols_dict, verbose=verbose)

    # 6. sampling by class
    if num_sample is not None:
        df = util.sampling_by_class(df, y_col, num_sample)

        # remove categorical cols with too few samples_per_cat

    return df, y_col


if __name__ == "__main__":
    df, y_col = load_obesity(verbose=True)
    df.info()
