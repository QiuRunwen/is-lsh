""" This module contains the functions to train the models."""

import logging
import os
import time

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn import metrics


import utils

CLASSIFIERS = (
    MLPClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    LogisticRegression,
    KNeighborsClassifier,
    SVC,
)

DICT_FUNCNAME_FUNC = {_cls.__name__: _cls for _cls in CLASSIFIERS}


def get_funcname_kwargs(func_names: list[str] = None) -> dict[str, dict]:
    """Get the name of functions and hyperparameters of the models."""
    if func_names is None:
        func_names = list(DICT_FUNCNAME_FUNC.keys())

    funcname_kwargs = {}
    for func_name in func_names:
        if func_name == "LinearSVC":
            # fix warning
            funcname_kwargs[func_name] = {"dual": "auto"}
        elif func_name == "DecisionTreeClassifier":
            funcname_kwargs[func_name] = {"max_depth": 30, "min_samples_split": 20}
        elif func_name == "RandomForestClassifier":
            funcname_kwargs[func_name] = {"max_depth": 30, "min_samples_split": 20}
        else:
            funcname_kwargs[func_name] = {}

    return funcname_kwargs


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    func_name: str,
    kwargs: dict = None,
    cache_dir: str = None,
) -> tuple[ClassifierMixin | RegressorMixin, float]:
    """Train a model with the given data and hyperparameters.

    Args:
        X_train (np.ndarray): The training data.
        y_train (np.ndarray): The training label.
        func_name (str): The name of the model which is in `d_funcname_func`.
        kwargs (dict, optional): The hyperparameters of the model. Defaults to None.
        cache_dir (str, optional): The cache dir. If not None, the result will be cached. Defaults to None.

    Returns:
        tuple[ClassifierMixin | RegressorMixin, float]: The trained model and the cpu time of training.
    """
    kwargs = kwargs if kwargs is not None else {}
    model = DICT_FUNCNAME_FUNC[func_name](**kwargs)
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        model_name = func_name + "_"
        for key, val in kwargs.items():
            model_name += f"{key}={val}_"
        model_name = model_name[:-1] + ".pkl"
        model_path = os.path.join(cache_dir, model_name)
        if os.path.exists(model_path):
            model, time_model = utils.load(model_path)
            logging.info("load from cache: %s", model_path)
        else:
            time_start = time.process_time()  # record the cpu time
            model.fit(X_train, y_train)
            time_model = time.process_time() - time_start

            utils.save((model, time_model), model_path)
            logging.info("save to cache: %s", model_path)

    else:
        time_start = time.process_time()  # record the cpu time
        model.fit(X_train, y_train)
        time_model = time.process_time() - time_start

    return model, time_model


def evaluate(
    model: ClassifierMixin | RegressorMixin,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Evaluate the model on the test set

    Args:
        model (ClassifierMixin | RegressorMixin): _description_
        X_test (np.ndarray): _description_
        y_test (np.ndarray): _description_

    Raises:
        ValueError: _description_

    Returns:
        dict: _description_
    """
    if isinstance(model, ClassifierMixin):  # classification
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = y_pred

        d_metric_val = {
            "Accuracy": metrics.accuracy_score(y_test, y_pred),
            "F1": metrics.f1_score(y_test, y_pred),
            "AUC": metrics.roc_auc_score(y_test, y_pred_proba),
            "Precision": metrics.precision_score(y_test, y_pred),
            "Recall": metrics.recall_score(y_test, y_pred),
        }

    elif isinstance(model, RegressorMixin):  # regression
        y_pred = model.predict(X_test)
        mse = metrics.mean_squared_error(y_test, y_pred)
        d_metric_val = {
            "MSE": mse,
            "RMSE": mse**0.5,
            "MAE": metrics.mean_absolute_error(y_test, y_pred),
            "MAPE": metrics.mean_absolute_percentage_error(y_test, y_pred),
            "R2": metrics.r2_score(y_test, y_pred),
            # 'MSLE': metrics.mean_squared_log_error(y_test, y_pred),
        }
    else:
        raise ValueError(
            f"model should be either BaseEstimator or RegressorMixin, got {type(model)}"
        )

    return d_metric_val
