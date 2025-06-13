"""
Banknote Authentication
https://archive.ics.uci.edu/dataset/267/banknote+authentication

Data were extracted from images that were taken for the evaluation of an authentication procedure for banknotes.

Data were extracted from images that were taken from genuine and forged banknote-like specimens.
For digitization, an industrial camera usually used for print inspection was used.
The final images have 400x 400 pixels. Due to the object lens and distance to the investigated
object gray-scale pictures with a resolution of about 660 dpi were gained.
Wavelet Transform tool were used to extract features from images. 

@author: QiuRunwen
"""

import os

import pandas as pd

if __name__ == "__main__":
    import util  # 单独运行本脚本用
else:
    from . import util  # 当作一个module被别人 import 时用


def load_banknote(
    *, data_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data"), drop_useless=True, num_sample=None, verbose=False
):
    # 1. read/uncompress data
    file_name = os.path.join(
        data_dir, "banknote+authentication/data_banknote_authentication.txt"
    )
    headers = ["variance", "skewness", "curtosis", "entropy", "class"]
    df = pd.read_csv(file_name, names=headers)

    #   Column    Non-Null Count  Dtype
    # ---  ------    --------------  -----
    # 0   variance  1372 non-null   float64   variance of Wavelet
    # 1   skewness  1372 non-null   float64   skewness of Wavelet
    # 2   curtosis  1372 non-null   float64   curtosis of Wavelet
    # 3   entropy   1372 non-null   float64   entropy of image
    # 4   class     1372 non-null   int64     class
    # dtypes: float64(4), int64(1)

    # 2. convert numeric/categorical columns

    # 3. simple feature extraction

    # 4. compute class label
    y_col = "class"
    #     class
    # 0    762
    # 1    610
    # Name: count, dtype: int64

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
    df, y_col = load_banknote(verbose=True)
    df.info()
