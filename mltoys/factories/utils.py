# mltoys/factories/utils.py

import random

from ..local import MLToyInstanceLocal


def get_feature_columns(d):
    return [f"feature_{i:d}" for i in range(d)]


def get_target_columns(d):
    return [f"target_{i:d}" for i in range(d)]


def make_instance(
    factory,
    seed,
    num_feature_columns,
    num_target_columns,
    loss_function,
    sample_function,
):
    feature_columns = get_feature_columns(num_feature_columns)
    target_columns = get_target_columns(num_target_columns)
    columns = ["id"] + feature_columns + target_columns

    training_data = [sample_function(i) for i in range(100)]
    test_data = [sample_function(i) for i in range(100, 200)]

    return MLToyInstanceLocal(
        factory=factory,
        seed=seed,
        columns=columns,
        feature_columns=feature_columns,
        target_columns=target_columns,
        loss_function=loss_function,
        training_data=training_data,
        test_data=test_data,
    )


def pick_seed(seed):
    if seed is None:
        seed = random.getrandbits(64)

    return seed


def sample_cube(r, d):
    return tuple(r.random() for _ in range(d))
