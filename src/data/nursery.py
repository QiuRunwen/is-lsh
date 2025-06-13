"""
Nursery Database was derived from a hierarchical decision model
originally developed to rank applications for nursery schools.
It was used during several years in 1980's when there was excessive enrollment
to these schools in Ljubljana, Slovenia, and the rejected applications
frequently needed an objective explanation.The final decision
depended on three subproblems: occupation of parents
and child's nursery, family structure and financial standing,
and social and health picture of the family. The model proposed should 
be able to predict whether the nursery is recommended for the child or not.

https://archive.ics.uci.edu/dataset/76/nursery

@author: QiuRunwen
"""

import os
from zipfile import ZipFile
import pandas as pd

if __name__ == "__main__":
    import util  # 单独运行本脚本用
else:
    from . import util  # 当作一个module被别人 import 时用


def load_nursery(
    data_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data"), drop_useless=True, num_sample=None, verbose=False
):
    # 1. read/uncompress data

    headers = [
        "parents",
        "has_nurs",
        "form",
        "children",
        "housing",
        "finance",
        "social",
        "health",
        "class",
    ]

    with ZipFile(os.path.join(data_dir, "nursery", "nursery.zip")) as zf:
        with zf.open("nursery.data") as f:  #
            df = pd.read_csv(f, header=None, names=headers, sep=",")

    # 12960*9

    # 2. convert numeric/categorical columns

    # parents        usual, pretentious, great_pret
    # has_nurs       proper, less_proper, improper, critical, very_crit
    # form           complete, completed, incomplete, foster
    # children       1, 2, 3, more
    # housing        convenient, less_conv, critical
    # finance        convenient, inconv
    # social         non-prob, slightly_prob, problematic
    # health         recommended, priority, not_recom

    # class         not_recom,recommend,very_recom,priority,spec_prior

    cat_cols = headers
    for col in cat_cols:
        if not isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype("category")

    # 3. simple feature extraction

    # 4. compute class label
    # class        N         N[%]
    # ------------------------------
    # not_recom    4320   (33.333 %)
    # recommend       2   ( 0.015 %)
    # very_recom    328   ( 2.531 %)
    # priority     4266   (32.917 %)
    # spec_prior   4044   (31.204 %)
    y_col = "class"
    df[y_col] = df[y_col].map(lambda health: 0 if health == "not_recom" else 1)
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
    df, y_col = load_nursery(verbose=True)
    df.info()
