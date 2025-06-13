"""test utils.py """
import os
import sys
import time
import unittest
from pathlib import Path

import numpy
import pandas as pd

current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent
sys.path.extend(
    [
        os.path.join(root_dir, "src"),
    ]
)

import utils


class TestUtils(unittest.TestCase):
    def test_seed_everything(self):
        """test utils.seed_everything"""
        utils.seed_everything(0)
        a = numpy.random.rand(10)
        utils.seed_everything(0)
        b = numpy.random.rand(10)
        self.assertTrue(numpy.allclose(a, b))
        utils.seed_everything(1)
        c = numpy.random.rand(10)
        self.assertFalse(numpy.allclose(a, c))

    def test_save_and_load(self):
        """test utils.save and utils.load"""
        obj = {"a": 1, "b": 2}
        for file_path in ["test.json", "test.yaml", "test.pkl", "test"]:
            utils.save(obj, file_path)
            obj_ = utils.load(file_path)
            self.assertEqual(obj, obj_)

        for file_path in ["test.json", "test.yaml", "test.pkl", "test"]:
            if os.path.exists(file_path):
                os.remove(file_path)

    def test_append_dicts_to_csv(self):
        """test utils.append_dicts_to_csv"""
        df = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
            }
        )

        file_path = "test.csv"

        # test file exits and have header
        df.to_csv(file_path, index=False)
        utils.append_dicts_to_csv(file_path=file_path, data_dicts=[{"a": 7, "b": 8}])
        utils.append_dicts_to_csv(
            file_path=file_path, data_dicts=[{"a": 9, "b": 10}, {"a": 11, "b": 12}]
        )

        # the keys of data_dicts must be the same as the header in the file
        utils.append_dicts_to_csv(
            file_path=file_path,
            data_dicts=[
                # {"a": 9},
                {"b": 12},
                {"b": 11, "a": 22},
                # {"c": 33, "a": 44},
                # {"d": 55, "e": 66},
            ],
        )

        # cannnot append dict with different keys
        # utils.append_dicts_to_csv(
        #     file_path=file_path,
        #     data_dicts=[{"a": 9, "b": 10, "c": "1"}, {"a": 11, "b": 12, "c": "2"}],
        # )

        # test empty file and have no header
        os.remove(file_path)
        with open(file_path, "w", newline="", encoding="utf-8") as _:
            pass
        utils.append_dicts_to_csv(file_path=file_path, data_dicts=[{"a": 7, "b": 8}])
        utils.append_dicts_to_csv(
            file_path=file_path, data_dicts=[{"a": 9, "b": 10}, {"a": 11, "b": 12}]
        )

        # finally remove the file
        os.remove(file_path)

    def test_timer_and_error_handler(self):
        """Normal case"""
        a = 1
        b = 2
        res_expected = a + b
        timeout = 3
        sleep_time = 1

        @utils.timer_and_error_handler(timeout=timeout)
        def func(a: int, b: int) -> int:
            time.sleep(sleep_time)
            return a + b

        res, excution_time, error = func(a, b)
        self.assertEqual(res, res_expected)
        self.assertLessEqual(excution_time, timeout)
        self.assertIsNone(error)

        def existed_func(a: int, b: int) -> int:
            time.sleep(sleep_time)
            return a + b

        new_existed_func = utils.timer_and_error_handler(timeout=timeout)(existed_func)
        res, excution_time, error = new_existed_func(a, b)
        self.assertEqual(res, res_expected)
        self.assertLessEqual(excution_time, timeout)
        self.assertIsNone(error)

    def test_timer_and_error_handler_2(self):
        """Timeout case"""
        a = 1
        b = 2
        # res_expected = a + b
        timeout = 1
        sleep_time = 3

        @utils.timer_and_error_handler(timeout=timeout)
        def func(a: int, b: int) -> int:
            time.sleep(sleep_time)
            return a + b

        res, excution_time, error = func(a, b)
        self.assertIsNone(res)
        self.assertGreaterEqual(excution_time, timeout)
        self.assertIsInstance(error, TimeoutError)

        def existed_func(a: int, b: int) -> int:
            time.sleep(sleep_time)
            return a + b

        new_existed_func = utils.timer_and_error_handler(timeout=timeout)(existed_func)
        res, excution_time, error = new_existed_func(a, b)
        self.assertIsNone(res)
        self.assertGreaterEqual(excution_time, timeout)
        self.assertIsInstance(error, TimeoutError)

    def test_timer_and_error_handler_3(self):
        """Raise exception case"""
        a = 1
        b = 2
        # res_expected = a + b
        timeout = 1

        @utils.timer_and_error_handler(timeout=timeout)
        def func(a: int, b: int) -> int:
            if a + b > 1:
                raise Exception("a+b>1")
            return a + b

        res, excution_time, error = func(a, b)
        self.assertIsNone(res)
        self.assertLessEqual(excution_time, timeout)
        self.assertIsInstance(error, Exception)

        def existed_func(a: int, b: int) -> int:
            if a + b > 1:
                raise MemoryError("a+b>1")
            return a + b

        new_existed_func = utils.timer_and_error_handler(timeout=timeout)(existed_func)
        res, excution_time, error = new_existed_func(a, b)
        self.assertIsNone(res)
        self.assertLessEqual(excution_time, timeout)
        self.assertIsInstance(error, MemoryError)


if __name__ == "__main__":
    unittest.main()
