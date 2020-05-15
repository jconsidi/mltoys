# mltoys/factories/linear.py

import random

from ..local import MLToyFactoryLocal
from .utils import make_instance
from .utils import pick_seed
from .utils import sample_cube


class LinearFunctionFactory(MLToyFactoryLocal):
    def get_instance(self, seed=None):
        seed = pick_seed(seed)

        r = random.Random(seed)

        d = random.randint(2, 11)
        coefficients = sample_cube(r, d)

        def sample(sample_id):
            x = sample_cube(r, d - 1)
            y = sum(c_i * x_i for (c_i, x_i) in zip(coefficients, x)) + coefficients[-1]

            return (sample_id,) + x + (y,)

        return make_instance(
            factory=self,
            seed=seed,
            num_feature_columns=d - 1,
            num_target_columns=1,
            loss_function="mean_squared_error",
            sample_function=sample,
        )
