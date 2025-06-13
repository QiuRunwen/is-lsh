"""Border Patterns LSH
Reproduce the paper: M. Aslani and S. Seipel, “Efficient and decision boundary aware instance selection for support vector machines,” 
    Information Sciences, vol. 577, pp. 579–598, Oct. 2021, doi: 10.1016/j.ins.2021.07.015.
    This code is rewritten based on the authors' MATLAB source code. https://github.com/mohaslani/BPLSH 
    
@author: QiuRunwen
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time


def BPLSH(
    Data: np.ndarray, M: int = 110, L: int = 50, W: float = 1, seed=None, alpha=None
) -> np.ndarray:
    """Border Patterns LSH.
    [1] M. Aslani and S. Seipel, “Efficient and decision boundary aware instance selection for support vector machines,”
    Information Sciences, vol. 577, pp. 579–598, Oct. 2021, doi: 10.1016/j.ins.2021.07.015.
    This code is rewritten based on the authors' MATLAB source code. https://github.com/mohaslani/BPLSH

    **Note**: In origin method, there is no parameter of `aplpha`.

    Args:
        Data (np.ndarray): The last col is label. Instances*(features+ label).
        M (int, optional): _description_. Defaults to 110.
        L (int, optional): _description_. Defaults to 50.
        W (float, optional): _description_. Defaults to 1.
        seed (_type_, optional): _description_. Defaults to None.
        alpha (_type_, optional): _description_. Defaults to None.

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

    # Generating the hash functions
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

    if np.max(Bucket_Index_Decimal_All) < 32767:
        Bucket_Index_Decimal_All = Bucket_Index_Decimal_All.astype(np.int16)

    # Instance Selection
    iii = 0  # matlab是1开始，python应该从0开始
    I = np.arange(Bucket_Index_Decimal_All.shape[1], dtype=np.int32)
    EP = []
    Bucket_Index_Decimal_All = Bucket_Index_Decimal_All.astype(np.int32)
    Classes = Data[:, -1]
    TRS = np.array([Bucket_Index_Decimal_All.shape[1] + 1], dtype=np.int32)
    Temporal_Removed_Samples = TRS
    Point_Extent = []
    Samples_OppositeClass_NearBoundary = np.array([], dtype=np.int32)
    SI = 2  # Similary Index

    while iii < I.size - 1:
        Current_Sample_Bucket_Index_Decimal = Bucket_Index_Decimal_All[:, iii].copy()
        Bucket_Index_Decimal_All[:, iii] = -1
        Number_of_Common_Buckets = np.sum(
            (
                Bucket_Index_Decimal_All
                - Current_Sample_Bucket_Index_Decimal.reshape((-1, 1))
            )
            == 0,
            axis=0,
        )
        Index_Neighbors = Number_of_Common_Buckets > 0
        Frequency_Neighbors = Number_of_Common_Buckets[Index_Neighbors]
        uniqued_Neighbors = I[Index_Neighbors]
        Bucket_Index_Decimal_All[:, iii] = Current_Sample_Bucket_Index_Decimal

        Class_Neighbors = Classes[uniqued_Neighbors]
        Class_Current = Classes[I[iii]]
        # if np.sum(np.diff(np.sort(np.hstack([Class_Neighbors, Class_Current]))) != 0) + 1 > 1:
        Classes_Neighbors_Unique = np.unique(Class_Neighbors)
        OppositeClasses = Classes_Neighbors_Unique[
            Classes_Neighbors_Unique != Class_Current
        ]
        if OppositeClasses.size != 0:  # 全部是否是同一个类，不是同一个类即 fully mixed
            # Fully Mixed
            # Classes_Neighbors_Unique = np.unique(Class_Neighbors)
            # OppositeClasses = Classes_Neighbors_Unique[Classes_Neighbors_Unique != Class_Current]
            SI_OppositeClasses = (
                Class_Neighbors == OppositeClasses.reshape(-1, 1)
            ) * Frequency_Neighbors.reshape(1, -1)
            Maximum_SI_OppositeClasses = np.max(SI_OppositeClasses, axis=1)
            Samples_OppositeClass_NearBoundary = np.hstack(
                [
                    Samples_OppositeClass_NearBoundary,
                    uniqued_Neighbors[
                        np.any(
                            SI_OppositeClasses
                            == Maximum_SI_OppositeClasses.reshape(-1, 1),
                            axis=0,
                        )
                    ],
                ]
            )
            if np.max(Maximum_SI_OppositeClasses) >= 1.0 * L:
                Samples_OppositeClass_NearBoundary = np.hstack(
                    [Samples_OppositeClass_NearBoundary, I[iii]]
                )
            else:
                Very_Close_Samples_SI = uniqued_Neighbors[
                    Frequency_Neighbors >= 1.0 * L
                ]
                Temporal_Removed_Samples = np.hstack(
                    [Temporal_Removed_Samples, Very_Close_Samples_SI]
                )
                # EP.append(I[iii])
                # Point_Extent.append(I[iii])
                # Just for making the algorithm fast
                if np.min(Temporal_Removed_Samples) <= I[iii + 1] or len(EP) > 2000:
                    mask = np.isin(I, np.hstack([Temporal_Removed_Samples, EP]))
                    I = I[~mask]
                    Bucket_Index_Decimal_All = Bucket_Index_Decimal_All[:, ~mask]
                    Temporal_Removed_Samples = TRS
                    iii -= len(EP)
                    EP.clear()
        else:
            # Unmixed
            Very_Close_Samples_SI = uniqued_Neighbors[Frequency_Neighbors >= SI]
            Temporal_Removed_Samples = np.hstack(
                [Temporal_Removed_Samples, Very_Close_Samples_SI]
            )
            EP.append(I[iii])
            Point_Extent.append(I[iii])
            # Just for making the algorithm fast
            if np.min(Temporal_Removed_Samples) <= I[iii + 1] or len(EP) > 2000:
                mask = np.isin(I, np.hstack([Temporal_Removed_Samples, EP]))
                I = I[~mask]
                Bucket_Index_Decimal_All = Bucket_Index_Decimal_All[:, ~mask]
                Temporal_Removed_Samples = TRS
                iii -= len(EP)
                EP.clear()

        iii += 1

    Selected_Data_Index = np.unique(
        np.hstack([Point_Extent, Samples_OppositeClass_NearBoundary])
    )

    if alpha is not None:
        # Selected_Data_Index = clip2raito(
        #     Data[:, -1], Selected_Data_Index, ratio=alpha, seed=seed
        # )
        num2select = int(alpha * Data.shape[0])
        curr_num = len(Selected_Data_Index)

        # curr_num should be less or equal to num2select
        if curr_num > num2select:
            Selected_Data_Index = rng.choice(
                Selected_Data_Index, num2select, replace=False
            )

    return raw_index[Selected_Data_Index]


# TODO. No use, to be deleted
# sampling usually require curr_num <= selection_num_required
def clip2raito(
    Data_labels: np.ndarray,
    Selected_Data_Index: np.ndarray,
    ratio: float = 1,
    seed=None,
) -> np.ndarray:
    """`Selected_Data_Index` has been sampled from the raw data. However, we need to
    1. keep the sample class-balanced.
    2. keep the final sample ratio equal to `ratio`.
    And now the class ratio corresponding to `Selected_Data_Index` may be too high or too low.

    Args:
        Data_labels (np.ndarray): The array containing all classes.
        Selected_Data_Index (np.ndarray): The array of indices selected from `Data_labels`.
        ratio (float): Desired final sample ratio, where 0 < ratio <= 1.

    Returns:
        np.ndarray: Final indexes selected.
    """
    if seed is None:
        seed = np.random.get_state()[1][0]  # get the seed of the global random state
    rng = np.random.default_rng(seed)

    if ratio >= 1:
        return Selected_Data_Index
    elif ratio <= 0:
        return np.ndarray([])
    else:
        labels, class_counts = np.unique(Data_labels, return_counts=True)
        num_should_selects = (
            (class_counts * ratio).astype(int) if ratio is not None else class_counts
        )
        # num_should_removes = class_counts - num_should_selects
        # d_label_shouldselect = dict(zip(labels, num_should_selects))
        remind_indexs = np.setdiff1d(
            np.arange(0, Data_labels.size), Selected_Data_Index
        )

        lables_selected = Data_labels[Selected_Data_Index]
        tmp_labels, tmp_label_selectcounts = np.unique(
            lables_selected, return_counts=True
        )
        d_label_selectcount = dict(zip(tmp_labels, tmp_label_selectcounts))

        final_indexs = []
        for label, num_should_select in zip(labels, num_should_selects):
            selectcount = d_label_selectcount.get(label, 0)
            curr_lable_index_selected = Selected_Data_Index[
                Data_labels[Selected_Data_Index] == label
            ]
            if selectcount < num_should_select:
                num_to_sample = num_should_select - selectcount
                label_remind_indexs = remind_indexs[Data_labels[remind_indexs] == label]
                new_sample_index = rng.choice(
                    label_remind_indexs, size=num_to_sample, replace=False
                )

                final_indexs.append(
                    Selected_Data_Index[Data_labels[Selected_Data_Index] == label]
                )
                final_indexs.append(new_sample_index)

            elif selectcount > num_should_select:
                # num_to_remove = selectcount - num_should_select
                new_sample_index = rng.choice(
                    curr_lable_index_selected, num_should_select, replace=False
                )
                final_indexs.append(new_sample_index)
            else:
                final_indexs.append(curr_lable_index_selected)

    return np.hstack(final_indexs)


def test_BPLSH_1(alpha=None, plot=True):
    np.set_printoptions(precision=15)

    # Reading Data
    Data = np.loadtxt("../../data/Sample-BPLSH.txt")

    # Simple Example
    M = 100  # 源码测试时是100
    L = 40  # 源码测试时是40
    W = 1
    Selected_Data_Index = BPLSH(Data, M, L, W, seed=1234, alpha=alpha)
    Selected_Data = Data[Selected_Data_Index, :]

    # Plotting Original and Selected Datasets
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.scatter(Data[:, 0], Data[:, 1], c=Data[:, 2], cmap="viridis", alpha=0.5)
        ax1.set(title="Original Datatset")
        ax2.scatter(
            Selected_Data[:, 0],
            Selected_Data[:, 1],
            c=Selected_Data[:, 2],
            cmap="viridis",
            alpha=0.5,
        )
        ax2.set(title="Selected Dataset")
        plt.suptitle(f"{M=},{L=},{W=},{Data.shape}->{Selected_Data.shape}")
        plt.show()


def test_BPLSH_2():
    Data = np.loadtxt("../../data/Sample-BPLSH.txt")

    W = 1
    L_Vector = [10, 20, 30]
    M_Vector = [20, 60, 100]
    TimeC = np.zeros((len(L_Vector), len(M_Vector), 10))
    Preserved_Size = np.zeros((len(L_Vector), len(M_Vector), 10))

    # For measuring Time and preservation rate
    for iteration in range(10):
        print(f"Iteration {iteration+1}")

        for i, L in enumerate(L_Vector):
            for j, M in enumerate(M_Vector):
                start_time = time()
                Selected_Index = BPLSH(Data, M, L, W)
                elapsed_time = time() - start_time

                TimeC[i, j, iteration] = elapsed_time
                Preserved_Size[i, j, iteration] = (
                    100 * len(Selected_Index) / Data.shape[0]
                )

    TimeC_Average = np.mean(TimeC, axis=2)
    Preserved_Size_Average = np.mean(Preserved_Size, axis=2)

    print("TimeC_Average:\n", TimeC_Average)
    print("\nPreserved_Size_Average:\n", Preserved_Size_Average)


def lsh_bp(
    X: np.ndarray,
    y: np.ndarray,
    *,
    w: float = 1,
    t: int = 110,
    L: int = 50,
    selection_rate: float = None,
    seed: int = None,
):
    """Data Reduction based on LSH considering the border pattern.
    [1] M. Aslani and S. Seipel, “Efficient and decision boundary aware instance selection for support vector machines,”
    Information Sciences, vol. 577, pp. 579–598, Oct. 2021, doi: 10.1016/j.ins.2021.07.015.
    This code is rewritten based on the authors' MATLAB source code. https://github.com/mohaslani/BPLSH

    Args:
        X (np.ndarray): The data matrix.
        y (np.ndarray): The label vector.
        w (float, optional): The bucket size. Defaults to 1.
        t (int, optional): The number of hash functions in each table. Defaults to 110.
        L (int, optional): The number of hash tables. Defaults to 50.
        selection_rate (float, optional): The selection rate. Defaults to None.
        seed (int, optional): The random seed. Defaults to None.

    Returns:
        np.ndarray: The selected indexes.
    """

    return BPLSH(
        np.hstack([X, y.reshape(-1, 1)]), M=t, L=L, W=w, seed=seed, alpha=selection_rate
    )


if __name__ == "__main__":
    test_BPLSH_1(plot=True, alpha=None)
    test_BPLSH_1(plot=True, alpha=0.1)
    test_BPLSH_1(plot=True, alpha=0.8)

    test_BPLSH_2()
