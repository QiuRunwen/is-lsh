"""
SMS Spam Collection Dataset
The SMS Spam Collection is a public set of SMS labeled messages that have been collected for mobile phone spam research.
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

@author: RunwenQiu
"""

import pandas as pd
import os

if __name__ == "__main__":
    import util  # 单独运行本脚本用
else:
    from . import util  # 当作一个module被别人 import 时用


def load_smsspam(
    data_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data"), drop_useless=True, num_sample=None, verbose=False
):
    # 1. read/uncompress data
    file_name = os.path.join(data_dir, "sms+spam+collection/SMSSpamCollection")
    df = pd.read_csv(file_name, header=0, names=["label", "message"], sep="\t")

    # 2. convert numeric/categorical columns

    # 3. simple feature extraction
    df["len_message"] = df["message"].str.len()
    df["contains_subscribe"] = (
        df["message"].str.contains("subscribe", case=False).astype(int)
    )
    df["contains_hash"] = df["message"].str.contains("#").astype(int)
    df["num_digits"] = df["message"].str.count(r"\d")
    df["contains_http"] = df["message"].str.contains("http", case=False).astype(int)
    # df["non_https"] =
    df["num_words"] = df["message"].str.count(r"\w+")
    df["contains_?"] = df["message"].str.contains(r"\?").astype(int)
    df["contains_www"] = df["message"].str.contains("www", case=False).astype(int)
    df["contains_money"] = df["message"].str.contains("money", case=False).astype(int)
    df["contains_free"] = df["message"].str.contains("free", case=False).astype(int)
    df["contains_win"] = df["message"].str.contains("win", case=False).astype(int)
    df["contains_call"] = df["message"].str.contains("call", case=False).astype(int)

    df.drop(columns="message", inplace=True)
    y_col = "label"
    df[y_col] = df["label"].transform(lambda x: 1 if x == "spam" else 0)

    # print(df.info())

    # 4. compute class label

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
    df, y_col = load_smsspam(verbose=True)
    df.info()
