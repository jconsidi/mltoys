# mltoys/factories/parity.py

import random

from ..local import MLToyFactoryLocal
from .utils import make_instance
from .utils import pick_seed


class ParityFunctionFactory(MLToyFactoryLocal):
    def get_instance(self, seed=None):
        seed = pick_seed(seed)

        r = random.Random(seed)

        d = random.randint(10, 20)

        bias = r.getrandbits(1)
        relevant = [r.getrandbits(1) for _ in range(d)]

        def sample(sample_id):
            bits = [r.getrandbits(1) for _ in range(d)]
            parity = (sum(r_i * b_i for (r_i, b_i) in zip(relevant, bits)) + bias) % 2

            return (sample_id,) + tuple(bits) + (parity,)

        return make_instance(
            factory=self,
            seed=seed,
            num_feature_columns=d,
            num_target_columns=1,
            loss_function="categorical_crossentropy",
            sample_function=sample,
        )
