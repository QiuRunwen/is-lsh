"""
Existing Synthetic data loading functions.
"""

import os
import pandas as pd

if __name__ == "__main__":
    import util  # 单独运行本脚本用
else:
    from . import util  # 当作一个module被别人 import 时用


def load_twonorm(
    *, data_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data"), drop_useless=True, num_sample=None, verbose=False
):
    """Leo Breiman's twonorm example[1]. It is a 20 dimensional, 2 class classification example
    [1] Breiman L. Bias, variance and arcing classifiers. Tec. Report 460, Statistics department.
    University of california. April 1996.
    https://www.cs.toronto.edu/~delve/data/twonorm/desc.html

    """
    # 1. read/uncompress data

    fp = os.path.join(data_dir, "twonorm/twonorm.csv")

    df = pd.read_csv(fp)

    # 7400 rows × (20+1) columns

    # 2. convert numeric/categorical columns
    # no need to convert

    # 3. simple feature extraction

    # 4. compute class label
    y_col = "Class"
    df[y_col] = df[y_col].transform(lambda x: 1 if x == 1 else 0)

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



def load_banana(
    *, data_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data"), drop_useless=True, num_sample=None, verbose=False
):
    """An artificial data set where instances belongs to several clusters with a banana shape.
    There are two attributes At1 and At2 corresponding to the x and y axis, respectively.
    https://sci2s.ugr.es/keel/dataset.php?cod=182#sub1
    """
    # 1. read/uncompress data

    fp = os.path.join(data_dir, "banana/banana.csv")

    df = pd.read_csv(fp)

    # 5300 rows × (2+1) columns

    # 2. convert numeric/categorical columns
    # no need to convert

    # 3. simple feature extraction

    # 4. compute class label
    y_col = "Class"
    df[y_col] = df[y_col].transform(lambda x: 1 if x == 1 else 0)

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
    for load in (load_twonorm, load_banana):
        df, y_col = load(verbose=True)
        df.info()
