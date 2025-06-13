"""
Reproduce the paper: M. Aslani and S. Seipel, “A fast instance selection method for support vector machines in building extraction,” 
    Applied Soft Computing, vol. 97, p. 106716, Dec. 2020, doi: 10.1016/j.asoc.2020.106716.
This code is rewritten based on the authors' MATLAB source code. https://github.com/mohaslani/DR.LSH

@author: QiuRunwen
"""

import numpy as np
import time
import matplotlib.pyplot as plt


def DRLSH(
    Data: np.ndarray,
    M: int = 20,
    L: int = 25,
    W: float = 1,
    ST: int = 7,
    seed: int = None,
    alpha: float = None,
) -> np.ndarray:
    """Data Reduction based on LSH.
    [1] M. Aslani and S. Seipel, “A fast instance selection method for support vector machines in building extraction,”
    Applied Soft Computing, vol. 97, p. 106716, Dec. 2020, doi: 10.1016/j.asoc.2020.106716.
    This code is rewritten based on the authors' MATLAB source code. https://github.com/mohaslani/DR.LSH

    **Note**: In origin method, there is no parameter of `aplpha`.

    Args:
        Data (np.ndarray): The last col is label. Instances*(features+ label).
        M (int): M times of LSH to construct g(x) = h_1(x), h_2(x), ..., h_m(x). Defaults to 20.
        L (int): L times of g(x). Defaults to 25.
        W (float): Parameters of LSH h(x;w). Defaults to 1.
        ST (int): Similarity index Threshhold. Defaults to 7.
        seed (int, optional): The random seed. Defaults to None.
        alpha (int, optional): The selected rate of dataset. If apha is not None, it will keep the class balanced.
            Defaults to None.

    Returns:
        np.ndarray: Selected_Data_Index
    """

    if seed is None:
        # current implementation is not thread-safe when seed is None
        seed = np.random.get_state()[1][0]  # get the seed of the global random state
    rng = np.random.default_rng(seed)

    # 算法和样本的顺序有关，所以要shuffle
    Data_Index = np.hstack([Data, np.arange(Data.shape[0]).reshape(-1, 1)])
    rng.shuffle(Data_Index)
    Data, raw_index = Data_Index[:, :-1], Data_Index[:, -1].astype(int)

    Classes, class_counts = np.unique(Data[:, -1], return_counts=True)

    num_should_selects = (
        (class_counts * alpha).astype(int) if alpha is not None else class_counts
    )
    num_should_removes = class_counts - num_should_selects

    # Normalizing the data
    maximum = np.max(Data[:, :-1], axis=0)
    minimum = np.min(Data[:, :-1], axis=0)
    maxmin = maximum - minimum
    maxmin[maxmin == 0] = 1
    Data[:, :-1] = (Data[:, :-1] - minimum) / maxmin

    Dimension = Data.shape[1] - 1  # Number of features
    M = int(M)  # Number of hash functions in each table
    L = int(L)  # Number of hash tables
    W = float(W)  # Bucket size

    # if the occurance frequency of a neighbor of the current point in all buckets is higher than this value,
    # it is removed. This value should be equal or less than L.
    Frequency_Neighbors_Threshold = int(ST)

    a = rng.normal(0, 1, [M * L, Dimension])  # Generate a in floor((ax+b)/W)
    b = W * rng.random([M * L, 1])  # Generate b in floor((ax+b)/W)
    # Calculating the buckets of samples
    Bucket_Index_Decimal_All = np.zeros([L, Data[:, :-1].shape[0]], dtype=np.int32)
    for i in range(L):
        j = np.arange((i * M), ((i + 1) * M))
        Bucket_Index = np.floor((a[j, :] @ Data[:, :-1].T + b[j, :]) / W).astype(
            np.int16
        )
        BI = Bucket_Index.T

        # For splitting BI matrix into partsNo to make the search faster
        Bucket_Index_uniqued = np.unique(BI, axis=0)

        # For splitting BI matrix into partsNo to make the search faster
        partsNo = 1
        ss = 0
        vectLength = BI.shape[1]
        splitsize = int(1 / partsNo * vectLength)
        for ij in range(partsNo):
            idxs = np.arange(
                int(np.round((ij - 1) * splitsize)), int(np.round(ij * splitsize))
            )
            BI_Part = BI[:, idxs]
            Bucket_Index_Decimal = np.apply_along_axis(
                lambda x: (
                    np.where((Bucket_Index_uniqued == x).all(axis=1))[0][0] + 1
                    if (Bucket_Index_uniqued == x).any()
                    else -1
                ),
                1,
                BI_Part,
            ).T
            Bucket_Index_Decimal_All[i, ss : ss + Bucket_Index_Decimal.shape[0]] = (
                Bucket_Index_Decimal
            )
            ss += Bucket_Index_Decimal.shape[0]

    # Removing samples
    # Removed_Samples_Index_ALL = np.zeros([Data[:, :-1].shape[0]], dtype=np.int32)
    Removed_Samples_Index_ALL = []
    RSC = 0  # remove sample counts

    for classID, num_should_remove in zip(Classes, num_should_removes):
        static_All_Indexes = np.where(Data[:, -1] == classID)[0]
        All_Indexes = static_All_Indexes.copy()
        Bucket_Index_Decimal_All_Class = Bucket_Index_Decimal_All[:, All_Indexes]
        iii = 0
        TRS = np.array([Bucket_Index_Decimal_All.shape[1] + 1], dtype=np.int32)
        Temporal_Removed_Samples = TRS

        # 遍历一个类下的所有变量
        while iii < All_Indexes.size - 1:
            # 获取邻居时移除自己
            Current_Sample_Bucket_Index_Decimal = Bucket_Index_Decimal_All_Class[
                :, iii
            ].copy()

            # bucketd_index是>=0的，所以这里相当于直接屏蔽了当前的bucketd_index是
            Bucket_Index_Decimal_All_Class[:, iii] = -1
            Number_of_Common_Buckets = np.sum(
                (
                    Bucket_Index_Decimal_All_Class
                    - Current_Sample_Bucket_Index_Decimal.reshape(-1, 1)
                )
                == 0,
                axis=0,
            )
            Index_Neighbors = Number_of_Common_Buckets > 0  # 邻居的index的filter

            # 邻居在同一个bucket的次数
            Frequency_Neighbors = Number_of_Common_Buckets[Index_Neighbors]
            uniqued_Neighbors = All_Indexes[Index_Neighbors]  # 邻居的index

            # 将之前赋-1时的还原
            Bucket_Index_Decimal_All_Class[:, iii] = Current_Sample_Bucket_Index_Decimal
            Removed_Samples_Current = uniqued_Neighbors[
                Frequency_Neighbors >= Frequency_Neighbors_Threshold
            ]
            # Removed_Samples_Index_ALL[RSC:RSC + Removed_Samples_Current.shape[0]] = Removed_Samples_Current
            Removed_Samples_Index_ALL.extend(Removed_Samples_Current)
            RSC += Removed_Samples_Current.shape[0]
            if alpha is not None and (RSC >= num_should_remove):
                break

            Temporal_Removed_Samples = np.hstack(
                [Temporal_Removed_Samples, Removed_Samples_Current]
            )
            # 如果需要remove的样本是已经遍历过的样本 或者 现在遍历次数已经超过2000了
            # 那么去除All_Indexes中已经remove的index，加快速度
            if (np.min(Temporal_Removed_Samples) <= All_Indexes[iii + 1]) or (
                iii > 2000
            ):
                aa = np.isin(All_Indexes, Temporal_Removed_Samples)
                All_Indexes = All_Indexes[~aa]
                Bucket_Index_Decimal_All_Class = Bucket_Index_Decimal_All_Class[:, ~aa]
                Temporal_Removed_Samples = TRS

                # Added
                All_Indexes = All_Indexes[iii + 1 :]
                Bucket_Index_Decimal_All_Class = Bucket_Index_Decimal_All_Class[
                    :, iii + 1 :
                ]
                iii = -1
            iii += 1

        if alpha is not None:
            if RSC < num_should_remove:
                remind_indexs = np.setdiff1d(
                    static_All_Indexes, Removed_Samples_Index_ALL
                )
                indexs_to_remove = rng.choice(
                    remind_indexs, num_should_remove - RSC, replace=False
                )
                Removed_Samples_Index_ALL.extend(indexs_to_remove)

    # Removed_Samples_Index_ALL = np.unique(Removed_Samples_Index_ALL)
    # Removed_Samples_Index_ALL = Removed_Samples_Index_ALL[Removed_Samples_Index_ALL != 0]
    Removed_Samples_Index_ALL = list(set(Removed_Samples_Index_ALL))

    Selected_Data_Index = np.setdiff1d(
        np.arange(Data.shape[0]), Removed_Samples_Index_ALL
    )

    return raw_index[Selected_Data_Index]


def test_DRLSH_1(plot=False, alpha=None):
    Data = np.loadtxt("../../data/Sample-DRLSH.txt", delimiter=",")
    # Data = np.array([
    #     [1,2,1]
    # ])

    # Simple Example
    ST = 6
    L = 10
    M = 25
    W = 1
    seed = 1234

    Selected_Index = DRLSH(Data, M, L, W, ST, seed, alpha=alpha)
    Selected_Data = Data[Selected_Index, :]

    # Plotting the original and selected dataset
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.scatter(Data[:, 0], Data[:, 1], c=Data[:, 2])
        ax1.set(title="Original Datatset")
        ax2.scatter(Selected_Data[:, 0], Selected_Data[:, 1], c=Selected_Data[:, 2])
        ax2.set(title="Selected Dataset")
        plt.suptitle(f"{M=},{L=},{W=},{Data.shape}->{Selected_Data.shape}")
        plt.show()


def test_DRLSH_2():
    # Calculating Tables 1 and 2 in the paper
    Data = np.loadtxt("../data/Sample-DRLSH.txt", delimiter=",")
    M_Vector = [30, 25, 20, 15, 10, 5]
    ST_Vector = [8, 6, 4, 2]
    W = 1
    Size_Vector_Time_Duration = []
    timeduration = []

    for iterations in range(100):
        print(f"Iteration: {iterations+1}")
        for i1, M in enumerate(range(len(M_Vector))):
            for ii1, ST in enumerate(range(len(ST_Vector))):
                L = 10
                W = 1

                t1 = time.time()
                Selected_Index_For_timeDuration = DRLSH(Data, M, L, W, ST)
                t2 = time.time()
                timeduration.append(t2 - t1)
                Size_Vector_Time_Duration.append(len(Selected_Index_For_timeDuration))

    timeduration = np.array(timeduration).reshape(
        (len(ST_Vector), len(M_Vector), iterations + 1)
    )
    Size_Vector_Time_Duration = np.array(Size_Vector_Time_Duration).reshape(
        (len(ST_Vector), len(M_Vector), iterations + 1)
    )

    print(np.mean(timeduration, axis=2))
    print(np.mean(Size_Vector_Time_Duration, axis=2))
    print(np.round(np.mean(timeduration, axis=2), 3))


def lsh_dr(
    X: np.ndarray,
    y: np.ndarray,
    *,
    w: float = 1,
    t: int = 20,
    L: int = 25,
    similarity_threshold: int = 7,
    selection_rate: float = None,
    seed: int = None,
) -> np.ndarray:
    """Data Reduction based on LSH.
    [1] M. Aslani and S. Seipel, “A fast instance selection method for support vector machines in building extraction,”
    Applied Soft Computing, vol. 97, p. 106716, Dec. 2020, doi: 10.1016/j.asoc.2020.106716.
    This code is rewritten based on the authors' MATLAB source code. https://github.com/mohaslani/DR.LSH

    Args:
        X (np.ndarray): _description_
        y (np.ndarray): _description_
        w (float, optional): _description_. Defaults to 1.
        t (int, optional): _description_. Defaults to 20.
        L (int, optional): _description_. Defaults to 25.
        similarity_threshold (int, optional): _description_. Defaults to 7.
        selection_rate (float, optional): _description_. Defaults to None.
        seed (int, optional): _description_. Defaults to None.

    Returns:
        np.ndarray: _description_
    """

    return DRLSH(
        Data=np.hstack([X, y.reshape(-1, 1)]),
        W=w,
        M=t,
        L=L,
        ST=similarity_threshold,
        alpha=selection_rate,
        seed=seed,
    )


if __name__ == "__main__":
    test_DRLSH_1(plot=True, alpha=None)
    test_DRLSH_1(plot=True, alpha=0.01)
    test_DRLSH_1(plot=True, alpha=0.8)
    # test_DRLSH_2()
