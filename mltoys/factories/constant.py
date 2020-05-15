# mltoys/factories/constant.py

import random

from ..local import MLToyFactoryLocal
from .utils import make_instance
from .utils import pick_seed
from .utils import sample_cube


class ConstantFunctionFactory(MLToyFactoryLocal):
    def get_instance(self, seed=None):
        seed = pick_seed(seed)

        r = random.Random(seed)

        d = random.randint(1, 10)
        mu = r.random()

        def sample(sample_id):
            return (sample_id,) + sample_cube(r, d) + (mu,)

        return make_instance(
            factory=self,
            seed=seed,
            num_feature_columns=d,
            num_target_columns=1,
            loss_function="mean_squared_error",
            sample_function=sample,
        )
