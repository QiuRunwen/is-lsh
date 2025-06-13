"""
Generate experiment config for exp.py and exp_pdoc.py
"""

import itertools

import datasets
import methods
import models
import utils


def _get_module_default_conf(
    module_name: str, ret_detail_default=True, handle_selrate=False
):
    if module_name == "datasets":
        if handle_selrate:
            raise ValueError("handle_selrate should be False for datasets")
        module_kwargs = datasets.get_funcname_kwargs()
    elif module_name == "methods":
        module_kwargs = methods.get_funcname_kwargs(handle_selrate=handle_selrate)
    elif module_name == "models":
        if handle_selrate:
            raise ValueError("handle_selrate should be False for models")
        module_kwargs = models.get_funcname_kwargs()
    else:
        raise ValueError(
            f"module_name: {module_name} not in ['datasets', 'methods', 'models']"
        )

    new_dict = {}
    for func_name, kwargs in module_kwargs.items():
        name = func_name
        new_dict[name] = {
            "func_name": func_name,
            "kwargs": kwargs.copy() if ret_detail_default else {},
        }

    return new_dict


def gen_default_exp_conf(save_path: str = None):
    """For exp.py"""
    conf_dict = {
        "dataset_func_kwargs": _get_module_default_conf("datasets"),
        "method_func_kwargs": _get_module_default_conf("methods", handle_selrate=True),
        "model_func_kwargs": _get_module_default_conf("models"),
        "other_kwargs": {
            "seeds": [0, 1, 2, 3, 4],
            "output_dir": "../output/result/default_exp_conf",
            "test_size": 0.2,
            "use_cache": False,
            "time_limit": 7200,
        },
    }

    if save_path is not None:
        utils.save(conf_dict, save_path)

    return conf_dict


def gen_1dnorma_pdoc_rnn(output_path: str):
    """Try to find the best pdoc partition in different dimensions and different sigma combinations in 1d normal distribution"""
    res_dict = {}
    width = 0.2
    # itervals = np.arange(-1, 1, width)
    # ranges = [(partition, partition + width) for partition in itervals]

    tmp_width = int(width * 10)
    ranges = [((i) / 10, (i + tmp_width) / 10) for i in range(-10, 10, tmp_width)]

    conbinations = itertools.product(ranges, ranges)
    for combination in conbinations:
        key = f"PDOC_1dnormal_{combination[0][0]:.2f}_{combination[0][1]:.2f}_{combination[1][0]:.2f}_{combination[1][1]:.2f}"
        res_dict[key] = {
            "func_name": "pdoc_r_nn",
            "kwargs": {"radius": 0, "thresholds": [combination[0], combination[1]]},
        }

    utils.save(res_dict, output_path)


def _gen_accurate_pdoc_partition(
    dimension: int,
    sigma: tuple[float, float],
    width: float = 0.2,
    output_dir: str = None,
    save_path: str = None,
):
    """generate accurate pdoc partition. For exp.py"""
    ls_n_samples = [
        [18000, 2000],
        [16000, 4000],
        [14000, 6000],
        [12000, 8000],
        [10000, 10000],
    ]

    res_dict = {
        "dataset_func_kwargs": {},
        "method_func_kwargs": {
            "noselect": {
                "func_name": "noselect",
                "kwargs": {},
            }
        },
        "model_func_kwargs": {
            "LR": {"func_name": "LogisticRegression", "kwargs": {}},
            "MLP": {"func_name": "MLPClassifier", "kwargs": {}},
            "RF": {"func_name": "RandomForestClassifier", "kwargs": {}},
            "SVM": {"func_name": "SVC", "kwargs": {}},
            "GBDT": {"func_name": "GradientBoostingClassifier", "kwargs": {}},
            "KNN": {"func_name": "KNeighborsClassifier", "kwargs": {}},
        },
        "other_kwargs": {
            "seeds": [0, 1, 2, 3, 4],
            "output_dir": (
                "../output/result/pdoc_distr" if output_dir is None else output_dir
            ),
            "test_size": 0.2,
            "use_cache": False,
        },
    }

    # width = 0.2
    # itervals = np.arange(-1, 1, width)
    # ranges = [(partition, partition + width) for partition in itervals]

    tmp_width = int(width * 10)
    ranges = [((i) / 10, (i + tmp_width) / 10) for i in range(-10, 10, tmp_width)]

    # conbinations = itertools.product(ranges, ranges)
    conbinations = list(itertools.combinations(ranges, 2))  # 会忽视自己和自己的组合
    for rgs in ranges:
        conbinations.append((rgs, rgs))

    for n_samples in ls_n_samples:
        dataset_key = f"gen_blobs_{dimension}_{n_samples[0]}_{n_samples[1]}_{sigma[0]:.2f}_{sigma[1]:.2f}"
        res_dict["dataset_func_kwargs"][dataset_key] = {
            "func_name": "gen_blobs",
            "kwargs": {"n_samples": n_samples, "sigma": sigma, "dimension": dimension},
        }

    for cbn in conbinations:
        for selection_rate in [None, 0.01]:
            method_key = f"pdoc_distr_{selection_rate}_{cbn[0][0]:.2f}_{cbn[0][1]:.2f}_{cbn[1][0]:.2f}_{cbn[1][1]:.2f}"
            res_dict["method_func_kwargs"][method_key] = {
                "func_name": "pdoc_gb",
                "kwargs": {
                    "thresholds": [cbn[0], cbn[1]],
                    "selection_rate": selection_rate,
                    "pdoc_as_weight": False,
                    "neg_mean": [-1] + [0] * (dimension - 1),
                    "neg_cov": sigma[0],
                    "pos_mean": [1] + [0] * (dimension - 1),
                    "pos_cov": sigma[1],
                },
            }

    if save_path is not None:
        utils.save(res_dict, save_path)
    return res_dict


def gen_find_best_pdoc():
    """Try to find the best pdoc partition in different dimensions and different sigma combinations.
    For exp.py
    """
    sigmas = [0.2, 0.5, 0.8, 1, 1.2, 1.5]
    dimensions = [1, 2, 3]
    sigma_combination = list(itertools.combinations(sigmas, 2))
    for sigma in sigmas:
        sigma_combination.append((sigma, sigma))

    for cbn in sigma_combination:
        for dimension in dimensions:
            _gen_accurate_pdoc_partition(
                dimension,
                cbn,
                width=0.2,
                output_dir=f"../output/result/blobs/blobs_{dimension}_{cbn[0]:.2f}_{cbn[1]:.2f}",
                save_path=f"../exp_conf/blobs/blobs_{dimension}_{cbn[0]:.2f}_{cbn[1]:.2f}.json",
            )


def gen_exp_pdoc(save_path: str = None):
    """For exp_pdoc.py"""
    conf_dict = {
        "dataset_func_kwargs": {},
        "method_func_kwargs": {
            "k_nn-use_weight=False": {
                "func_name": "k_nn",
                "kwargs": {"n_neighbors": 30, "use_weight": False},
            },
            "k_nn-use_weight=True": {
                "func_name": "k_nn",
                "kwargs": {"n_neighbors": 30, "use_weight": True},
            },
            "r_nn-use_weight=False": {
                "func_name": "r_nn",
                "kwargs": {"radius": 0.1, "use_weight": False},
            },
            "r_nn-use_weight=True": {
                "func_name": "r_nn",
                "kwargs": {"radius": 0.1, "use_weight": True},
            },
            "lsh-use_weight=False": {
                "func_name": "lsh",
                "kwargs": {"w": 1, "t": 5, "L": 10, "use_weight": False},
            },
            "lsh-use_weight=True": {
                "func_name": "lsh",
                "kwargs": {"w": 1, "t": 5, "L": 10, "use_weight": True},
            },
        },
        "other_kwargs": {
            "seeds": [0, 1, 2, 3, 4],
            "output_dir": "../output/result/estim_pdoc",
            "test_size": None,
            "use_cache": False,
        },
    }

    ls_n_samples = [
        [18000, 2000],
        [16000, 4000],
        [14000, 6000],
        [12000, 8000],
        [10000, 10000],
    ]
    sigmas = [0.2, 0.5, 0.8, 1, 1.2, 1.5]
    sigma_combination = list(itertools.product(sigmas, sigmas))
    dimensions = [1, 2, 3]

    for n_samples, sigma, dimension in itertools.product(
        ls_n_samples, sigma_combination, dimensions
    ):
        dataset_key = f"gen_blobs_{dimension}_{n_samples[0]}_{n_samples[1]}_{sigma[0]:.2f}_{sigma[1]:.2f}"
        conf_dict["dataset_func_kwargs"][dataset_key] = {
            "func_name": "gen_blobs",
            "kwargs": {"n_samples": n_samples, "sigma": sigma, "dimension": dimension},
        }
    if save_path is not None:
        utils.save(conf_dict, save_path)


def gen_delete_miscls():
    """Find what happens when we delete misclassified samples. For exp.py"""
    sigmas = [0.2, 0.5, 0.8, 1, 1.2, 1.5]
    dimensions = [1, 2, 3]
    sigma_combination = list(itertools.combinations(sigmas, 2))
    for sigma in sigmas:
        sigma_combination.append((sigma, sigma))

    for cbn in sigma_combination:
        for dimension in dimensions:
            _gen_accurate_pdoc_partition(
                dimension,
                cbn,
                width=1,
                output_dir=f"../output/result/delete_misclf/blobs_{dimension}_{cbn[0]:.2f}_{cbn[1]:.2f}",
                save_path=f"../exp_conf/delete_misclf/blobs_{dimension}_{cbn[0]:.2f}_{cbn[1]:.2f}.json",
            )


def get_diff_selrate(
    sel_rates: list[float] = None,
    include_noselect: bool = True,
    specific_funcname: str = None,
    specific_kwargs: dict = None,
    methods2pop: list[str] = None,
):
    """Get method config with different selection rate.
    Args:
        sel_rates: selection rate for methods. If None, use [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.8]
        only support .2f
        specific_funcname: if not None, only generate config for this method
        specific_kwargs: if not None, use this kwargs for specific_funcname
    Returns:
        method_func_kwargs: dict, e.g. `{"k_nn_0.01": {"func_name": "k_nn", "kwargs": {"selection_rate": 0.01}}}`
    """
    if sel_rates is None:
        sel_rates = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.8]

    dict_funcname_kwargs = methods.get_funcname_kwargs(handle_selrate=False)

    methods2pop = (
        [
            # "NE",
            # "PDOC_Voronoi",
            "pdoc_knn",
            "pdoc_rnn",
            "pdoc_gb",
        ]
        if methods2pop is None
        else methods2pop
    )
    methods2pop.append("noselect")
    for m in methods2pop:
        dict_funcname_kwargs.pop(m)

    method_func_kwargs = {}
    if include_noselect:
        method_func_kwargs["noselect"] = {
            "func_name": "noselect",
            "kwargs": {},
        }

    for selection_rate in sel_rates:
        for funcname, kwargs in dict_funcname_kwargs.items():
            if specific_funcname is not None:
                if funcname != specific_funcname:
                    continue
                if specific_kwargs is not None:
                    kwargs = specific_kwargs

            tmp_kwargs = kwargs.copy()
            tmp_kwargs["selection_rate"] = selection_rate
            method_func_kwargs[f"{funcname}_{selection_rate:.2f}"] = {
                "func_name": funcname,
                "kwargs": tmp_kwargs,
            }
    return method_func_kwargs


def gen_synthetic_ds(save_path: str = None):
    """For exp.py"""
    dict_conf = {
        "dataset_func_kwargs": {
            "load_twonorm": {"func_name": "load_twonorm", "kwargs": {}},
            "load_banana": {"func_name": "load_banana", "kwargs": {}},
        },
        "method_func_kwargs": get_diff_selrate(),
        "model_func_kwargs": _get_module_default_conf(module_name="models"),
        "other_kwargs": {
            "seeds": [0, 1, 2, 3, 4],
            "output_dir": "../output/result/current/synthetic_ds",
            "test_size": 0.2,
            "use_cache": False,
            "time_limit": 7200,
        },
    }

    # funcnames = ["gen_blobs", "gen_circles", "gen_moons", "gen_xor"]

    # different sigma
    for sigma in [0.2, 0.8, 1.5]:
        tmp = {
            f"gen_blobs_5000_{sigma:.2f}_2": {
                "func_name": "gen_blobs",
                "kwargs": {"n_samples": 5000, "sigma": sigma, "dimension": 2},
            },
            f"gen_circles_5000_{sigma:.2f}": {
                "func_name": "gen_circles",
                "kwargs": {"n_samples": 5000, "sigma": sigma},
            },
            f"gen_moons_5000_{sigma:.2f}": {
                "func_name": "gen_moons",
                "kwargs": {"n_samples": 5000, "sigma": sigma},
            },
            f"gen_xor_5000_{sigma:.2f}": {
                "func_name": "gen_xor",
                "kwargs": {"n_samples": 5000, "sigma": sigma},
            },
        }
        dict_conf["dataset_func_kwargs"].update(tmp)

    # different dimension for blobs
    for dimension in [2, 4, 6, 8, 16, 32, 64, 128, 256, 512]:
        dict_conf["dataset_func_kwargs"][f"gen_blobs_5000_0.80_{dimension}"] = {
            "func_name": "gen_blobs",
            "kwargs": {"n_samples": 5000, "sigma": 0.8, "dimension": dimension},
        }

    # different n_samples for blobs , 50000, 100000
    for n_samples in [1000, 2000, 5000, 10000, 20000]:
        dict_conf["dataset_func_kwargs"][f"gen_blobs_{n_samples}_0.80_2"] = {
            "func_name": "gen_blobs",
            "kwargs": {"n_samples": n_samples, "sigma": 0.8, "dimension": 2},
        }

    if save_path is not None:
        utils.save(dict_conf, save_path)

    return dict_conf


def gen_natural_ds(save_path: str = None):
    """For exp.py."""
    dict_conf = {
        "dataset_func_kwargs": {
            "load_mammographic": {"func_name": "load_mammographic", "kwargs": {}},
            "load_banknote": {"func_name": "load_banknote", "kwargs": {}},
            "load_obesity": {"func_name": "load_obesity", "kwargs": {}},
            "load_smsspam": {"func_name": "load_smsspam", "kwargs": {}},
            "load_mushroom": {"func_name": "load_mushroom", "kwargs": {}},
            "load_nursery": {"func_name": "load_nursery", "kwargs": {}},
        },
        "method_func_kwargs": get_diff_selrate(),
        "model_func_kwargs": _get_module_default_conf(module_name="models"),
        "other_kwargs": {
            "seeds": [0, 1, 2, 3, 4],
            "output_dir": "../output/result/current/natural_ds",
            "test_size": 0.2,
            "use_cache": False,
            "time_limit": 7200,
        },
    }
    if save_path is not None:
        utils.save(dict_conf, save_path)

    return dict_conf


def gen_lsh_ours(save_path: str = None):
    dict_conf = gen_natural_ds()
    dict_conf2 = gen_synthetic_ds()
    dict_conf["dataset_func_kwargs"].update(dict_conf2["dataset_func_kwargs"])

    dict_conf["other_kwargs"]["output_dir"] = "../output/result/lsh_ours_param"
    dict_conf["method_func_kwargs"] = {
        "noselect": {
            "func_name": "noselect",
            "kwargs": {},
        }
    }

    # sts = ("near_random", "near_filter", "far_random", "far_filter", "all_random")
    sts = ("near_random", "near_filter", "all_random")
    srs = (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.8)
    uws = (True, False)
    ths = [(0, 1), (0.1, 1), (0.2, 1)]
    for st, sr, uw, th in itertools.product(sts, srs, uws, ths):
        dict_conf["method_func_kwargs"][f"lsh_ours_{st}_{sr:.2f}_{uw}_{th}"] = {
            "func_name": "lsh_ours",
            "kwargs": {
                "thresholds": [th],
                "selection_rate": sr,
                "selection_type": st,
                "use_weight": uw,
                "use_op_arg": True,
            },
        }

    if save_path is not None:
        utils.save(dict_conf, save_path)

    return dict_conf


def gen_diff_pdoc_inblob(save_path: str = None, sigma=0.8, dimension=2):

    dict_conf = {
        "dataset_func_kwargs": {
            f"gen_blobs_5000_{sigma:.2f}_{dimension}": {
                "func_name": "gen_blobs",
                "kwargs": {"n_samples": 5000, "sigma": sigma, "dimension": dimension},
            }
        },
        "method_func_kwargs": {
            "noselect": {
                "func_name": "noselect",
                "kwargs": {},
            }
        },
        "model_func_kwargs": _get_module_default_conf(module_name="models"),
        "other_kwargs": {
            "seeds": [0, 1, 2, 3, 4],
            "output_dir": f"../output/result/pdoc_inblob/{sigma:.2f}_{dimension}",
            "test_size": 0.2,
            "use_cache": False,
            "time_limit": 7200,
        },
    }

    for st in (
        "near_random",
        "near_filter",
        "far_random",
        "far_filter",
        "all_random",
    ):
        for sr in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.8]:
            dict_conf["method_func_kwargs"][f"pdoc_gb_{st}_{sr:.2f}"] = {
                "func_name": "pdoc_gb",
                "kwargs": {
                    "thresholds": [(0, 1)],
                    "selection_rate": sr,
                    "selection_type": st,
                    "neg_mean": [-1] + [0] * (dimension - 1),
                    "neg_cov": sigma,
                    "pos_mean": [1] + [0] * (dimension - 1),
                    "pos_cov": sigma,
                    "pos_proba_prior": 0.5,
                },
            }

    if save_path is not None:
        utils.save(dict_conf, save_path)

    return dict_conf


def gen_diff_pdoc_innat(save_path: str = None):

    dict_conf = gen_natural_ds()
    dict_conf["other_kwargs"]["output_dir"] = "../output/result/pdoc_innat"
    dict_conf["method_func_kwargs"] = {
        "noselect": {
            "func_name": "noselect",
            "kwargs": {},
        }
    }

    for st in (
        "near_random",
        "near_filter",
        "far_random",
        "far_filter",
        "all_random",
    ):
        for sr in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.8]:
            # TODO KNN may be not accurate to estimate the pdoc in natural dataset
            dict_conf["method_func_kwargs"][f"pdoc_knn_{st}_{sr:.2f}"] = {
                "func_name": "pdoc_knn",
                "kwargs": {
                    "thresholds": [(0, 1)],
                    "selection_rate": sr,
                    "selection_type": st,
                    "n_neighbors": 30,
                },
            }

    if save_path is not None:
        utils.save(dict_conf, save_path)

    return dict_conf


def gen_onemethod(save_path: str = None, ds_type: str = "synthetic", method="noselect"):

    # synthetic
    if ds_type == "synthetic":
        conf = gen_synthetic_ds()
        conf["other_kwargs"]["output_dir"] = f"../output/result/{method}/synthetic_ds"
    elif ds_type == "natural":
        conf = gen_natural_ds()
        conf["other_kwargs"]["output_dir"] = f"../output/result/{method}/natural_ds"
    else:
        raise ValueError(f"ds_type: {ds_type} not in ['synthetic', 'natural']")

    if method == "noselect":
        conf["method_func_kwargs"] = {
            "noselect": {"func_name": "noselect", "kwargs": {}}
        }
    else:
        conf["method_func_kwargs"] = get_diff_selrate(
            include_noselect=False, specific_funcname=method
        )

    if save_path is not None:
        utils.save(conf, save_path)

    return conf


if __name__ == "__main__":

    print(gen_default_exp_conf("../exp_conf/default_exp_conf.json"))

    print(gen_synthetic_ds("../exp_conf/synthetic_ds.json"))

    print(gen_natural_ds("../exp_conf/natural_ds.json"))


    # print(gen_exp_pdoc("../exp_conf/exp_pdoc.json"))
