# mltoys/factories/linear.py

import random

from ..local import MLToyFactoryLocal
from ..local import MLToyInstanceLocal
from .utils import get_feature_columns
from .utils import get_target_columns
from .utils import pick_seed
from .utils import sample_cube


class LinearFunctionFactory(MLToyFactoryLocal):
    def get_instance(self, seed=None):
        seed = pick_seed(seed)

        r = random.Random(seed)

        d = random.randint(2, 11)
        coefficients = sample_cube(r, d)

        feature_columns = get_feature_columns(d - 1)  # skip bias/constant
        target_columns = get_target_columns(1)

        def sample(sample_id):
            x = sample_cube(r, d - 1)
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
