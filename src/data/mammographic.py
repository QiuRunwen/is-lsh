# -*- coding: utf-8 -*-
"""
Mammographic Mass
Discrimination of benign and malignant mammographic masses based on BI-RADS attributes and the patient's age.
https://archive.ics.uci.edu/dataset/161/mammographic+mass

@author: QiuRunwen
"""

import os
import pandas as pd

if __name__ == "__main__":
    import util  # 单独运行本脚本用
else:
    from . import util  # 当作一个module被别人 import 时用


def load_mammographic(
    data_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data"), drop_useless=True, num_sample=None, verbose=False
):
    # 1. read/uncompress data

    file_path = os.path.join(
        data_dir, "mammographic+mass", "mammographic_masses.data"
    )
    names = ["BI-RADS assessment", "age", "shape", "margin", "density", "severity"]
    df = pd.read_csv(file_path, header=None, names=names, na_values="?")

    # Number of Instances: 961
    # Number of Attributes: 6 (1 goal field, 1 non-predictive, 4 predictive attributes)
    # Attribute Information:
    # 1. BI-RADS assessment: 1 to 5 (ordinal)
    # 2. Age: patient's age in years (integer)
    # 3. Shape: mass shape: round=1 oval=2 lobular=3 irregular=4 (nominal)
    # 4. Margin: mass margin: circumscribed=1 microlobulated=2 obscured=3 ill-defined=4 spiculated=5 (nominal)
    # 5. Density: mass density high=1 iso=2 low=3 fat-containing=4 (ordinal)
    # 6. Severity: benign=0 or malignant=1 (binominal)

    # 2. convert numeric/categorical columns

    cat_cols = [col for col in df.columns if col not in ["age", "severity"]]
    for col in cat_cols:
        if not isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype("category")

    # 3. simple feature extraction

    # 4. compute class label
    # Class Distribution: benign: 516; malignant: 445

    y_col = "severity"
    assert df[y_col].notna().all()

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
    df, y_col = load_mammographic(verbose=True)
    df.info()
