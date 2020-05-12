# mltoys/loss_functions.py

from math import inf
from math import log

_loss_functions = {}


def _register(f):
    _loss_functions[f.__name__] = f
    return f


def check_loss_function(loss_function):
    return loss_function in _loss_functions


def get_loss_function(loss_function):
    return _loss_functions.get(loss_function)


############################################################
# actual loss functions ####################################
############################################################


@_register
def categorical_crossentropy(expected_values, actual_values):
    """
    Compute cross entropy between actual and expected predicted values.

    Individual losses are capped to avoid 
    """

    if len(expected_values) != len(actual_values):
        raise RuntimeError("length mismatch")

    return sum(
        [
            e * (-log(a) if a > 0 else inf)
            for (e, a) in zip(expected_values, actual_values)
            if e != 0
        ],
        0.0,
    )


@_register
def mean_squared_error(expected_values, actual_values):
    if len(expected_values) != len(actual_values):
        raise RuntimeError("length mismatch")

    return sum((e - a) ** 2 for (e, a) in zip(expected_values, actual_values)) / len(
        expected_values
    )
