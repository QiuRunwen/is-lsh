import json
import logging
import math
import os
import sys
from pathlib import Path
import warnings
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# add exeutable path `../../Library` of glpk and ipopt to PATH
# visible in this process + all children
curr_file_path = Path(__file__)
os.environ["PYOMO_CONFIG_DIR"] = os.path.abspath(
    os.path.join(curr_file_path.parent.parent.parent.resolve(), "Library")
)

import pyomo.environ as pyo
import seaborn as sns
from scipy import integrate, optimize, special

def lsh_2stable(X: np.ndarray, w: float = 1, t: int = 1, L: int = 1, seed: int = None):
    """2-stable Local sensitive hash.

    Args:
        X (np.ndarray): Datasets with shape (size, dimension)
        w (float, optional): parameter of pstable-LSH. Defaults to 1.
        t (int, optional): Times of lsh. g(x)=(h_1(x),...,h_t(x)). Defaults to 1.
        L (int, optional): Number of hash tables. Defaults to 1.
        seed (int, optional): Random seed to reproduce results.

    Returns:
        List[np.ndaaray]: [HashTable1, HashTable2,..,HashTableL].
    """
    if w <= 0 or t <= 0 or L <= 0:
        raise ValueError(
            f"`w`, `t`, `L` should be all greater than 0. Now they are {w,t,L}"
        )

    if seed is not None:
        np.random.seed(seed)
    n, dimension = X.shape

    # TODO 这里没有考虑后续有新的点进来，所以暂时不保存`np.random.randn(d,t)`和`np.random.random(t)`
    return [
        np.floor((X @ np.random.randn(dimension, t) / w + np.random.random(t)))
        for _ in range(L)
    ]

def convert_hashtable(
    hashtables: list[np.ndarray],
    ret_counter: bool = False,
) -> tuple[list[dict], list[list[set]]]:
    """Convert hashtable to `hashvalue_i:bucket_i` like `{g_1: [pIndex1,pIndexk,...], g_2:[pIndex2,...],... }`

    Args:
        hashtables (list[np.ndarray]): _description_

    Returns:
        tuple[list[dict], list[list[set]]]:
        ls_dict : `[{g_1: [pIndex1,pIndexk,...], g_2:[pIndex2,...],... }, {...},... ]`
        ls_buckets: `[ [[pIndex1,pIndexk,...],[pIndex2,pIndexk,...],...],  [...],...  ]`
    """

    # TODO 进一步加速
    L = len(hashtables)  # 表的个数
    n = len(hashtables[0])  # 数据集的size
    ls_dict = [
        defaultdict(list) for _ in range(L)
    ]  # 注意这里的dict不要指向同一个了. `[{}]*3`就会指向同一个dict
    ls_buckets = [[None] * n for _ in range(L)]
    
    ls_dict_counter = None
    ls_counters = None
    if ret_counter:
        
        # np.uint32, 4294967295
        def my_counter():
            return np.zeros(n, dtype=np.uint32)
        ls_dict_counter = [defaultdict(my_counter) for _ in range(L)]
        ls_counters = [[None] * n for _ in range(L)]
    
    for i, hashtable in enumerate(hashtables): # i=0,1,..,L-1
        d_hash_bucket = ls_dict[i]
        buckets = ls_buckets[i]
        
        if ret_counter:
            d_hash_counter = ls_dict_counter[i]
            counters = ls_counters[i]
        
        for j, arr in enumerate(hashtable): # j=0,1,..,n-1
            # https://zhuanlan.zhihu.com/p/454916593 默认是按行一依次取bytes
            tup = arr.tobytes()
            tmp_bucket = d_hash_bucket[tup]
            tmp_bucket.append(j)
            # new_table[j] = tup
            buckets[j] = tmp_bucket
            
            if ret_counter:
                tmp_counter = d_hash_counter[tup]
                tmp_counter[j] += 1
                counters[j] = tmp_counter
            

    # TODO 以后跑更大一点的数据时，可以不用返回ls_dict
    return ls_dict, ls_buckets, ls_dict_counter, ls_counters

def count_idx_in_bucket(ls_buckets: list[list[set]]) -> list[Counter]:
    size = len(ls_buckets[0])  # 数据集大小
    ls_d_idx_count = [None for _ in range(size)]

    for i in range(size):
        idxs_collision = [idx for buckets in ls_buckets for idx in buckets[i]]
        d_idx_count = Counter(idxs_collision)
        del d_idx_count[i] #= 0  # 排除自己
        ls_d_idx_count[i] = d_idx_count

    return ls_d_idx_count

# def merge_counters(ls_counters: list[list[Counter]]) -> list[Counter]:
#     """merge counters in ls_counters to one Counter for each element in ls_counters"""
#     return [sum(counters, Counter()) for counters in ls_counters]

def merge_counters(ls_counters: list[list[np.ndarray]]) -> list[np.ndarray]:
    """merge counters in ls_counters to one Counter for each element in ls_counters"""
    return [np.sum(counters, axis=0) for counters in ls_counters]


import timeit
X = np.random.randn(20000, 2)
w = 0.01
t = 1
L = 100
ret_counter = False
hashtables = lsh_2stable(X,w,t,L)
ls_dict, ls_buckets, ls_dict_counter, ls_counters = convert_hashtable(hashtables, ret_counter=ret_counter)
# print(len(ls_counters))
# print(len(ls_counters[0]))
# ls_d_idx_count = count_idx_in_bucket(ls_buckets)
# print(timeit.timeit('lsh_2stable(X,w,t,L)', globals=globals(), number=1))
print(timeit.timeit(f'convert_hashtable(hashtables, ret_counter={ret_counter})', globals=globals(), number=1))
# print(timeit.timeit('merge_counters(ls_counters)', globals=globals(), number=1)) # Counter相加太慢了
print(timeit.timeit('count_idx_in_bucket(ls_buckets)', globals=globals(), number=1))