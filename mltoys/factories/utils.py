# mltoys/factories/utils.py

import random


def get_feature_columns(d):
    return [f"feature_{i:d}" for i in range(d)]


def get_target_columns(d):
    return [f"target_{i:d}" for i in range(d)]


def pick_seed(seed):
    if seed is None:
        seed = random.getrandbits(64)

    return seed


def sample_cube(r, d):
    return tuple(r.random() for _ in range(d))
