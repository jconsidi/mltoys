#!/usr/bin/env python3

import sys

from catboost import CatBoostRegressor

import mltoys


from base import SklearnBase


class CatBoostModel(SklearnBase):
    def __init__(self, columns, feature_columns, target_columns, loss_function):
        super().__init__(
            columns=columns,
            feature_columns=feature_columns,
            target_columns=target_columns,
            loss_function=loss_function,
        )

        self.model = CatBoostRegressor(
            depth=min(6, len(feature_columns)), loss_function="RMSE", silent=True,
        )


def main():
    print("catboost model:")

    for factory in mltoys.get_factories():
        if factory.__class__.__name__.startswith("Constant"):
            continue

        mltoys.test_models(model_class=CatBoostModel, factory=factory)

    return 0


############################################################
# startup handling #########################################
############################################################

if __name__ == "__main__":
    sys.exit(main())
