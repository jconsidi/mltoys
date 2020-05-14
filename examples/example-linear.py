#!/usr/bin/env python3

import sys

from sklearn.linear_model import LinearRegression

import mltoys

from base import SklearnBase


class LinearModel(SklearnBase):
    def __init__(self, columns, feature_columns, target_columns, loss_function):
        super().__init__(
            columns=columns,
            feature_columns=feature_columns,
            target_columns=target_columns,
            loss_function=loss_function,
        )

        self.model = LinearRegression()


def main():
    print("linear model:")

    for factory in mltoys.get_factories():
        mltoys.test_models(model_class=LinearModel, factory=factory)

    return 0


############################################################
# startup handling #########################################
############################################################

if __name__ == "__main__":
    sys.exit(main())
