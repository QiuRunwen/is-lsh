"""Experiment pipeline."""

import itertools
import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

import utils
from datasets import load_one_assign_name
from methods import sample_data
from models import train_model, evaluate
from preprocess import split_impute_encode_scale


def check_exp_id_executed(
    exp_id: str, executed_exp_ids: list[str], *tup_upcoming_args: list[str]
) -> bool:
    """Check if the experiment has been executed.

    Args:
        exp_id (str): Current experiment id.
        executed_exp_ids (list[str]): List of executed experiment ids.
        *tup_upcoming_args (list[str]): List of upcoming arguments.

    Returns:
        bool: _description_
    """
    if tup_upcoming_args:
        upcoming_exp_ids = [
            "-".join(arg_tuple)
            for arg_tuple in itertools.product([exp_id], *tup_upcoming_args)
        ]
    else:
        upcoming_exp_ids = [exp_id]

    # `all(e in ls2 for e in ls1)`, a generator expression is used to create an iterator that yields True or False.
    # `all([e in ls2 for e in ls1])`, a list comprehension is used instead of a generator expression
    #  first method using a generator expression is more memory-efficient
    return all(
        upcoming_exp_id in executed_exp_ids for upcoming_exp_id in upcoming_exp_ids
    )


def exp(
    dataset_func_kwargs: dict[str, dict],
    method_func_kwargs: dict[str, dict],
    model_func_kwargs: dict[str, dict],
    test_size: float = 0.2,
    seeds: list[int] = None,
    output_dir: str = "./output",
    use_cache: bool = False,
    min_samples_per_class: int = 30,
    time_limit: int = 3600,
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
    error_path = os.path.join(output_dir, "error.csv")
    toofew_path = os.path.join(output_dir, "toofew.csv")
    df_result = (
        pd.read_csv(result_path) if os.path.exists(result_path) else pd.DataFrame()
    )

    executed_exp_ids = (
        df_result["exp_id"].unique().tolist() if "exp_id" in df_result.columns else []
    )
    error_exp_ids = (
        pd.read_csv(error_path)["exp_id"].unique().tolist()
        if os.path.exists(error_path)
        else []
    )
    toofew_exp_ids = (
        pd.read_csv(toofew_path)["exp_id"].unique().tolist()
        if os.path.exists(toofew_path)
        else []
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
                model_func_kwargs.keys(),
            ):
                logging.info("exp_id: %s has been executed, skip", exp_id_seed_dataset)
                continue

            dataset_func_name = tmp_dict["func_name"]
            dataset_kwargs = tmp_dict["kwargs"]

            utils.seed_everything(seed)
            dataset = load_one_assign_name(
                func_name=dataset_func_name, kwargs=dataset_kwargs, name=dataset_name
            )

            X_train, X_test, y_train, y_test = split_impute_encode_scale(
                df=dataset.df,
                y_col=dataset.y_col,
                test_size=test_size,
                seed=seed,
            )

            for method_name, tmp_kwargs in tqdm(
                method_func_kwargs.items(), desc=exp_id_seed_dataset
            ):
                exp_id_seed_dataset_method = exp_id_seed_dataset + "-" + method_name
                if exp_id_seed_dataset_method in error_exp_ids:
                    logging.info(
                        "method: %s with error, skip", exp_id_seed_dataset_method
                    )
                    continue

                if exp_id_seed_dataset_method in toofew_exp_ids:
                    logging.info(
                        "method: %s contains too few samples, skip",
                        exp_id_seed_dataset_method,
                    )
                    continue

                if check_exp_id_executed(
                    exp_id_seed_dataset_method,
                    executed_exp_ids,
                    model_func_kwargs.keys(),
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
                tmp_func = utils.timer_and_error_handler(timeout=time_limit)(
                    sample_data
                )
                result, execute_time, error = tmp_func(
                    X=X_train,
                    y=y_train,
                    func_name=method_func_name,
                    kwargs=method_kwargs,
                    cache_dir=method_cache_dir,
                )
                is_exp_success = error is None
                if is_exp_success:
                    # pylint: disable=unpacking-non-sequence
                    X_train_sub, y_train_sub, time_method = result
                else:
                    logging.error(
                        "exp_id: %s failed. %s",
                        exp_id_seed_dataset_method,
                        error,
                    )

                    tmp_dict_res = {
                        "exp_id": exp_id_seed_dataset_method,
                        "seed": seed,
                        "dataset": dataset_name,
                        "method": method_name,
                        "execute_time": execute_time,
                        "raw_train_size": X_train.shape[0],
                        "test_size": X_test.shape[0],
                        "train_dim": X_train.shape[1],
                        "error": error.__class__.__name__,
                    }
                    utils.append_dicts_to_csv(error_path, [tmp_dict_res])
                    error_exp_ids.append(exp_id_seed_dataset_method)
                    continue

                # check if the sub dataset contains too few samples
                # each class should have enough samples for model training
                y_sub_unique, y_sub_counts = np.unique(y_train_sub, return_counts=True)
                if (
                    y_sub_unique.size != 2
                    or y_sub_counts.size == 0
                    or (np.min(y_sub_counts) < min_samples_per_class)
                ):
                    utils.append_dicts_to_csv(
                        toofew_path,
                        [
                            {
                                "exp_id": exp_id_seed_dataset_method,
                                "seed": seed,
                                "dataset": dataset_name,
                                "method": method_name,
                                "time_method": time_method,
                                "raw_train_size": X_train.shape[0],
                                "train_size": X_train_sub.shape[0],
                                "test_size": X_test.shape[0],
                                "train_dim": X_train.shape[1],
                            }
                        ],
                    )
                    toofew_exp_ids.append(exp_id_seed_dataset_method)
                    logging.warning(
                        "%s contains too few samples, skip. %s, %s",
                        exp_id_seed_dataset_method,
                        y_sub_unique,
                        y_sub_counts,
                    )
                    continue

                for model_name, tmp_kwargs in tqdm(
                    model_func_kwargs.items(), desc=exp_id_seed_dataset_method
                ):
                    exp_id_seed_dataset_method_model = (
                        exp_id_seed_dataset_method + "-" + model_name
                    )
                    if check_exp_id_executed(
                        exp_id_seed_dataset_method_model, executed_exp_ids
                    ):
                        logging.info(
                            "exp_id: %s has been executed, skip",
                            exp_id_seed_dataset_method_model,
                        )
                        continue

                    model_func_name = tmp_kwargs["func_name"]
                    model_kwargs = tmp_kwargs["kwargs"]
                    model_cache_dir = None
                    if use_cache:
                        model_cache_dir = os.path.join(
                            output_dir,
                            "cache",
                            exp_id_seed_dataset_method_model.replace("-", "/"),
                        )

                    utils.seed_everything(seed)
                    model, time_model = train_model(
                        func_name=model_func_name,
                        X_train=X_train_sub,
                        y_train=y_train_sub,
                        kwargs=model_kwargs,
                        cache_dir=model_cache_dir,
                    )
                    dict_res = {
                        "exp_id": exp_id_seed_dataset_method_model,
                        "seed": seed,
                        "dataset": dataset_name,
                        "method": method_name,
                        "time_method": time_method,
                        "model": model_name,
                        "time_model": time_model,
                        "raw_train_size": X_train.shape[0],
                        "train_size": X_train_sub.shape[0],
                        "test_size": X_test.shape[0],
                        "train_dim": X_train.shape[1],
                    }

                    dict_res.update(evaluate(model, X_test, y_test))
                    result_dicts.append(dict_res)

                    utils.append_dicts_to_csv(result_path, [dict_res])
                    executed_exp_ids.append(exp_id_seed_dataset_method_model)

    df_result = pd.concat([df_result, pd.DataFrame(result_dicts)], axis=0)

    return df_result


def execute_exp(path: str):
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
                model_func_kwargs=exp_dict["model_func_kwargs"],
                **exp_dict["other_kwargs"],
            )
    else:
        raise ValueError(f"Invalid path: {path}")


def _test():
    dict_dataset_func_kwargs = {
        "moons_0.1": {
            "func_name": "gen_moons",
            "kwargs": {"n_samples": 1000, "sigma": 0.2},
        },
    }

    dict_method_func_kwargs = {
        # "RandomSampling": {
        #     "func_name": "RandomSampling",
        #     "kwargs": {"random_state": 0},
        # },
        "PDOC_kNN": {
            "func_name": "pdoc_knn",
            "kwargs": {"n_neighbors": 1, "selection_rate": 0.1, "seed": 0},
        }
    }

    dict_model_func_kwargs = {
        "LR": {
            "func_name": "LogisticRegression",
            "kwargs": {},
        }
    }

    other_kwargs = {
        "seeds": [0, 1, 2, 3, 4],
        "output_dir": "../output/result/test",
        "test_size": 0.2,
        "use_cache": True,
    }

    exp(
        dict_dataset_func_kwargs,
        dict_method_func_kwargs,
        dict_model_func_kwargs,
        **other_kwargs,
    )


def main(exp_file_path=None):
    """Priority: cmd > exp_file_path > test"""
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
    main(
        "../exp_conf/natural_ds.json"
        # "../exp_conf/synthetic_ds.json"
        # "../exp_conf/default_exp_conf.json"
        # "../exp_conf/blobs"
        # "../exp_conf/delete_misclf"
        # "../exp_conf/lsh_ours.json"
    )
