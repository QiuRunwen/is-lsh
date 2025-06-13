"""
The utils function used in src
"""

import csv
import functools
import inspect
import json
import logging
import os
import pickle
import random
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# TODO np.random.default_rng and np.random.RandomState are not controlled by seed_everything
def seed_everything(seed: int):
    """seed all random function"""
    if seed is None:
        return

    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save(obj: object, file_path: str):
    """save object to file_path as pickle file or json file or yaml file"""

    name, extension = os.path.splitext(file_path)
    if not extension:
        extension = ".pkl"
        file_path = name + extension

    supported_extensions = [".yaml", ".json", ".pkl"]

    # check extension
    if extension not in supported_extensions:
        raise ValueError(f"extension: {extension} not in {supported_extensions}")

    # create dir
    dir_name = os.path.dirname(file_path)
    if len(dir_name) > 0:
        os.makedirs(dir_name, exist_ok=True)

    # save
    if extension == ".json":
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(obj, file)
    elif extension == ".pkl":
        with open(file_path, "wb") as file:
            pickle.dump(obj, file)
    elif extension == ".yaml":
        # avoid reference cycle
        # yaml.Dumper.ignore_aliases = lambda *args : True
        with open(file_path, "w", encoding="utf-8") as file:
            yaml.dump(obj, file)
    else:
        raise ValueError(f"file_extension: {extension} not in {supported_extensions}")


def load(file_path: str):
    """load object from pickle file or json file"""

    supported_extensions = [".yaml", ".json", ".pkl"]

    # check file_path
    name, _ = os.path.splitext(file_path)
    if not os.path.exists(file_path):
        for supported_extension in supported_extensions:
            tmp_file_path = name + supported_extension
            if os.path.exists(tmp_file_path):
                file_path = tmp_file_path
                break

    # load object
    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as file:
            obj = json.load(file)
    elif file_path.endswith(".pkl"):
        with open(file_path, "rb") as file:
            obj = pickle.load(file)
    elif file_path.endswith(".yaml"):
        with open(file_path, "r", encoding="utf-8") as file:
            obj = yaml.load(file, Loader=yaml.FullLoader)
    else:
        raise ValueError(f"file_path: {file_path} not end with {supported_extension}")

    return obj


def dict_to_json(dict_: dict):
    """convert dict to json"""
    return json.dumps(dict_)


def append_dicts_to_csv(file_path: str | Path, data_dicts: list[dict]):
    """Append dicts to csv file. If the file is empty, write header using the keys of data_dicts.
    if the file is not empty, the first line of the file must be the header.
    and the keys of data_dicts must be the same as the header in the file.
    e.g. data_dicts=[{"a": 1, "b": 2}, {"a": 3, "b": 4}] and the header is ["a", "b"]
    """
    if not data_dicts:
        return  # Return if the list is empty

    # Create the file and directory if not exist
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "a+", newline="", encoding="utf-8") as csv_file:
        if csv_file.tell() == 0:
            # The file is empty, write header
            fieldnames = data_dicts[0].keys()
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
        else:
            # The file is not empty, get the header first and then write
            csv_file.seek(0)  # Move the cursor to the start of the file
            fieldnames = csv.DictReader(csv_file).fieldnames  # Read the header
            csv_file.seek(0, 2)  # Move the cursor to the end of the file
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerows(data_dicts)


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self, *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


def timer_and_error_handler(timeout=None):
    """A decorator to set timeout for a function. Return the result, execution time and error.
    If the function is not finished in the given time, it will be stopped.
    If the function raises an exception, the exception will be caught and returned,
    result will be None.
    Usage example:
    ```python
    @timer_and_error_handler(timeout=10)
    def func(a,b):
        return a+b

    result, execution_time, error = func(1,2)


    def existed_func(a,b):
        return a+b

    new_exsited_func = timer_and_error_handler(timeout=10)(existed_func)
    result, execution_time, error = func(1,2)
    ```
    """

    def decorator(func):
        @functools.wraps(func)  # keep the original function's name and docstring
        def wrapper(
            *args, **kwargs
        ) -> tuple[None | Any, float, None | TimeoutError | MemoryError | Exception]:
            result = [None]
            execution_time = [None]
            error = [None]
            start_time = time.time()

            def target():
                try:
                    # start_time = time.time()
                    result[0] = func(*args, **kwargs)
                    # end_time = time.time()
                    # execution_time[0] = end_time - start_time
                except MemoryError as e:
                    error[0] = e
                except TimeoutError as e:
                    error[0] = e
                except Exception as e:
                    error[0] = e

            # windows can not use `signal` library. Use threading instead.
            # https://stackoverflow.com/questions/52779920/why-is-signal-sigalrm-not-working-in-python-on-windows
            # signal.signal(signal.SIGALRM, _timeout_handler)

            thread = StoppableThread(target=target)
            thread.start()
            thread.join(timeout)

            if thread.is_alive():  # timeout
                # _timeout_handler(func, timeout)
                logging.warning(
                    "Function `%s` execution timed out after %d seconds.",
                    func.__name__,
                    timeout,
                )
                error[0] = TimeoutError("TimeoutError")
                # stop the thread after timeout
                thread.stop()

            end_time = time.time()
            execution_time[0] = end_time - start_time
            return result[0], execution_time[0], error[0]

        return wrapper

    return decorator


def get_function_default_params(func):
    """Get the default parameters of a function.
    Args:
        func (function): The function to get default parameters.
    Returns:
        tuple[dict, dict]: The default parameters and required parameters.
    """

    signature = inspect.signature(func)
    param_default = {}
    param_required = {}

    for param_name, param in signature.parameters.items():
        if param.default != inspect.Parameter.empty:
            param_default[param_name] = param.default
        else:
            param_required[param_name] = param.default

    return param_default, param_required
