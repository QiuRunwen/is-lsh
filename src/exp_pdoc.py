"""Experiment pipeline for estimating pdoc."""

import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

import utils
from datasets import load_one_assign_name
from estim_pdoc import estimate_pdoc
from exp import check_exp_id_executed
from ismethod.pdoc import eval_estimate
from ismethod.utils import format_y_binary


def exp(
    dataset_func_kwargs: dict[str, dict],
    method_func_kwargs: dict[str, dict],
    test_size: float = None,
    seeds: list[int] = None,
    output_dir: str = "./output",
    use_cache: bool = False,
):
    """experiment pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    logging.captureWarnings(True)
    logging.basicConfig(
        filename=os.path.join(output_dir, "exp.log"),
        level=logging.INFO,
        format="%(asctime)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s",
        encoding="utf-8",
    )
    logging.info("------start exp------")
    if seeds is None:
        warnings.warn("seeds is None, will use a random int as seed")
        seeds = [np.random.randint(0, 10000)]

    result_path = os.path.join(output_dir, "result.csv")
    df_result = (
        pd.read_csv(result_path) if os.path.exists(result_path) else pd.DataFrame()
    )

    executed_exp_ids = (
        df_result["exp_id"].unique().tolist() if "exp_id" in df_result.columns else []
    )

    result_dicts = []
    for seed in tqdm(seeds, desc="seed"):
        if seed is None:
            seed = np.random.randint(0, 10000)
            utils.seed_everything(seed)

        exp_id_seed = str(seed)
        for dataset_name, tmp_dict in tqdm(
            dataset_func_kwargs.items(), desc=exp_id_seed
        ):
            exp_id_seed_dataset = exp_id_seed + "-" + dataset_name
            if check_exp_id_executed(
                exp_id_seed_dataset,
                executed_exp_ids,
                method_func_kwargs.keys(),
            ):
                logging.info("exp_id: %s has been executed, skip", exp_id_seed_dataset)
                continue

            dataset_func_name = tmp_dict["func_name"]
            dataset_kwargs = tmp_dict["kwargs"]

            utils.seed_everything(seed)
            dataset = load_one_assign_name(
                func_name=dataset_func_name, kwargs=dataset_kwargs, name=dataset_name
            )
            pdocs = dataset.get_pdoc()
            if test_size is None:
                df, y_col = dataset.df, dataset.y_col
                df[y_col] = format_y_binary(df[y_col])
                X_test, y_test = (
                    df[df.columns[df.columns != y_col]].to_numpy(),
                    df[y_col].to_numpy(),
                )
                pdoc_test = pdocs
                X_train, y_train = None, None

                X_test = StandardScaler().fit_transform(X_test)
            else:
                raise NotImplementedError

            for method_name, tmp_kwargs in tqdm(
                method_func_kwargs.items(), desc=exp_id_seed_dataset
            ):
                exp_id_seed_dataset_method = exp_id_seed_dataset + "-" + method_name
                if check_exp_id_executed(
                    exp_id_seed_dataset_method,
                    executed_exp_ids,
                ):
                    logging.info(
                        "exp_id: %s has been executed, skip", exp_id_seed_dataset_method
                    )
                    continue

                method_func_name = tmp_kwargs["func_name"]
                method_kwargs = tmp_kwargs["kwargs"]
                method_cache_dir = None
                if use_cache:
                    method_cache_dir = os.path.join(
                        output_dir,
                        "cache",
                        exp_id_seed_dataset_method.replace("-", "/"),
                    )

                utils.seed_everything(seed)
                pdoc_test_estim, time_method = estimate_pdoc(
                    X=X_test,
                    y=y_test,
                    func_name=method_func_name,
                    kwargs=method_kwargs,
                    cache_dir=method_cache_dir,
                )

                dict_res = {
                    "exp_id": exp_id_seed_dataset_method,
                    "seed": seed,
                    "dataset": dataset_name,
                    "method": method_name,
                    "time_method": time_method,
                    # "train_size": X_train_sub.shape[0],
                    "test_size": X_test.shape[0],
                    "test_dim": X_test.shape[1],
                    # "train_dim": X_train.shape[1],
                }

                dict_res.update(eval_estimate(pdoc_test, pdoc_test_estim))
                result_dicts.append(dict_res)

                utils.append_dicts_to_csv(result_path, [dict_res])
                executed_exp_ids.append(exp_id_seed_dataset_method)

    df_result = pd.concat([df_result, pd.DataFrame(result_dicts)], axis=0)

    return df_result


def execute_exp(path: str):
    path = os.path.abspath(path)
    if os.path.isdir(path):
        for file in os.listdir(path):
            execute_exp(os.path.join(path, file))
    elif os.path.isfile(path):
        if path.endswith(".json") or path.endswith(".yaml"):
            exp_dict = utils.load(path)
            utils.save(
                exp_dict,
                os.path.join(exp_dict["other_kwargs"]["output_dir"], "conf.json"),
            )
            exp(
                dataset_func_kwargs=exp_dict["dataset_func_kwargs"],
                method_func_kwargs=exp_dict["method_func_kwargs"],
                **exp_dict["other_kwargs"],
            )
    else:
        raise ValueError(f"Invalid path: {path}")


def _test():
    dict_dataset_func_kwargs = {
        "blobs": {
            "func_name": "gen_blobs",
            "kwargs": {"n_samples": [1000, 1000], "sigma": [0.8, 0.8], "dimension": 2},
        },
    }

    dict_method_func_kwargs = {
        "k_nn": {
            "func_name": "k_nn",
            "kwargs": {"n_neighbors": 2, "use_weight": False},
        },
        "r_nn": {
            "func_name": "r_nn",
            "kwargs": {"radius": 0.1, "use_weight": False},
        },
        "lsh": {
            "func_name": "lsh",
            "kwargs": {"w": 1, "t": 5, "L": 10, "use_weight": False},
        },
    }

    other_kwargs = {
        "seeds": [0, 1, 2, 3, 4],
        "output_dir": "../output/result/test_estim",
        "test_size": None,
        "use_cache": True,
    }

    exp(
        dict_dataset_func_kwargs,
        dict_method_func_kwargs,
        **other_kwargs,
    )


def main(exp_file_path=None):
    if exp_file_path is not None:
        sys.argv.append(exp_file_path)

    if len(sys.argv) < 2:
        print("Usage: python exp.py [exp_file_path]")
        print("No exp_file_path is given, use test()")
        _test()
    else:
        exp_file_path = sys.argv[1]
        execute_exp(exp_file_path)


if __name__ == "__main__":
    main("../exp_conf/estim_pdoc/20230721.json")
