#!/usr/bin/env python3

"""
Unit tests for local problems.
"""

import math
import unittest

from mltoys import MLToyInstanceLocal


class InstanceLocalTestMixin:
    def check_loss(self, test_outputs, expected_loss):
        actual_loss = self.instance.score(test_outputs)
        self.assertAlmostEqual(actual_loss, expected_loss)


class MeanSquaredErrorTestCase(InstanceLocalTestMixin, unittest.TestCase):
    def setUp(self):
        self.instance = MLToyInstanceLocal(
            factory=None,
            seed=None,
            columns=["id", "a", "b", "c"],
            feature_columns=["a", "b"],
            target_columns=["c"],
            loss_function="mean_squared_error",
            training_data=[],
            test_data=[[1, 2, 3, 4], [5, 6, 7, 8]],
        )

    def test_00(self):
        self.check_loss([[1, 4], [5, 8]], 0.0)
        self.check_loss([[1, 6], [5, 6]], 4.0)
        self.check_loss([[1, 6], [5, 8]], 2.0)


class CategoryCrossentropyTestCase(InstanceLocalTestMixin, unittest.TestCase):
    def setUp(self):
        # targets are even/odd parity
        self.instance = MLToyInstanceLocal(
            factory=None,
            seed=None,
            columns=["id", "a", "b", "c", "d"],
            feature_columns=["a", "b"],
            target_columns=["c", "d"],
            loss_function="categorical_crossentropy",
            training_data=[],
            test_data=[
                [1, 0, 0, 1, 0],
                [2, 0, 1, 0, 1],
                [3, 1, 0, 0, 1],
                [4, 1, 1, 1, 0],
            ],
        )

    def test_00(self):
        self.check_loss(
            [[1, 1, 0], [2, 0, 1], [3, 0, 1], [4, 1, 0]], 0.0
        )  # perfect answer
        self.check_loss(
            [[1, 0.9, 0.1], [2, 0.1, 0.9], [3, 0.1, 0.9], [4, 0.9, 0.1]], -math.log(0.9)
        )  # confident answers
        self.check_loss(
            [[1, 0.5, 0.5], [2, 0.5, 0.5], [3, 0.5, 0.5], [4, 0.5, 0.5]], -math.log(0.5)
        )  # coin flip
        self.check_loss(
            [[1, 0.1, 0.9], [2, 0.9, 0.1], [3, 0.9, 0.1], [4, 0.1, 0.9]], -math.log(0.1)
        )  # bad answers
        self.check_loss(
            [[1, 0, 1], [2, 1, 0], [3, 1, 0], [4, 0, 1]], math.inf
        )  # completely wrong answers


############################################################
# startup handling #########################################
############################################################

if __name__ == "__main__":
    unittest.main()
