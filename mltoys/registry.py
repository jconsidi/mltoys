# mltoys/registry.py

from typing import Iterable

from . import factories
from .types import MLToyFactory

_factories = []


def get_factories() -> Iterable[MLToyFactory]:
    for f in _factories:
        yield f


def register_factory(factory_class, **kwargs):
    """
    Class decorator that creates an instance of the class and saves it
    for future calls to get_factories().
    """

    _factories.append(factory_class(**kwargs))
    return factory_class


############################################################
# explicitly populate the local factories ##################
############################################################

register_factory(factories.ConstantFunctionFactory)
register_factory(factories.LinearFunctionFactory)
register_factory(factories.ParityFunctionFactory)
