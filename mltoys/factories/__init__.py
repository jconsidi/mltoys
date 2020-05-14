# mltoys/factories.py

import random

from ..local import MLToyFactoryLocal
from ..local import MLToyInstanceLocal


def _feature_columns(d):
    return [f"feature_{i:d}" for i in range(d)]


def _pick_seed(seed):
    if seed is None:
        seed = random.getrandbits(64)

    return seed


def _sample_cube(r, d):
    return tuple(r.random() for _ in range(d))


def _target_columns(d):
    return [f"target_{i:d}" for i in range(d)]


class ConstantFunctionFactory(MLToyFactoryLocal):
    def get_instance(self, seed=None):
        seed = _pick_seed(seed)

        r = random.Random(seed)

        d = random.randint(1, 10)
        mu = r.random()

        feature_columns = _feature_columns(d)
        target_columns = _target_columns(1)

        def sample(sample_id):
            return (sample_id,) + _sample_cube(r, d) + (mu,)

        columns = ["id"] + feature_columns + target_columns
        training_data = [sample(i) for i in range(100)]
        test_data = [sample(i) for i in range(100, 200)]

        return MLToyInstanceLocal(
            factory=self,
            seed=seed,
            columns=columns,
            feature_columns=feature_columns,
            target_columns=target_columns,
            loss_function="mean_squared_error",
            training_data=training_data,
            test_data=test_data,
        )


class LinearFunctionFactory(MLToyFactoryLocal):
    def get_instance(self, seed=None):
        seed = _pick_seed(seed)

        r = random.Random(seed)

        d = random.randint(2, 11)
        coefficients = _sample_cube(r, d)

        feature_columns = _feature_columns(d - 1)  # skip bias/constant
        target_columns = _target_columns(1)

        def sample(sample_id):
            x = _sample_cube(r, d - 1)
            y = sum(c_i * x_i for (c_i, x_i) in zip(coefficients, x)) + coefficients[-1]

            return (sample_id,) + x + (y,)

        columns = ["id"] + feature_columns + target_columns
        training_data = [sample(i) for i in range(100)]
        test_data = [sample(i) for i in range(100, 200)]

        return MLToyInstanceLocal(
            factory=self,
            seed=seed,
            columns=columns,
            feature_columns=feature_columns,
            target_columns=target_columns,
            loss_function="mean_squared_error",
            training_data=training_data,
            test_data=test_data,
        )
