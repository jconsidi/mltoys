# mltoys/factories/circle.py

import math
import random

from ..local import MLToyFactoryLocal
from .utils import make_instance
from .utils import pick_seed
from .utils import sample_cube


class CircleFunctionFactory(MLToyFactoryLocal):
    def get_instance(self, seed=None):
        seed = pick_seed(seed)

        r = random.Random(seed)

        d = 2
        center = sample_cube(r, d)
        radius = 0.125 + random.random() * 0.125

        def sample(sample_id):
            x = sample_cube(r, 2)
            y = 1 if math.sqrt((x[0] - center[0])**2 + (x[1] - center[1])**2) <= radius else 0

            return (sample_id,) + x + (y,)

        return make_instance(
            factory=self,
            seed=seed,
            num_feature_columns=d,
            num_target_columns=1,
            loss_function="categorical_crossentropy",
            sample_function=sample,
        )
