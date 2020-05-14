#!/usr/bin/env python3

import sys

from catboost import CatBoostRegressor

import mltoys


class CatBoostModel(mltoys.types.MLToyBase):
    def __init__(self, columns, feature_columns, target_columns, loss_function):
        super().__init__(
            columns=columns,
            feature_columns=feature_columns,
            target_columns=target_columns,
            loss_function=loss_function,
        )

        self.model = CatBoostRegressor(
            depth=min(6, len(feature_columns)), loss_function="RMSE",
            silent=True,
        )

    def fit(self, training_data):
        """
        Records distinct values from target columns.
        """

        training_data = tuple(map(tuple, training_data))

        num_target_columns = len(self.target_columns)
        self.model.fit(
            [r[1:-num_target_columns] for r in training_data],
            [r[-num_target_columns:] for r in training_data],
        )

    def predict(self, test_data):
        """
        Pick a random value recorded for each column.
        """

        test_data = tuple(map(tuple, test_data))

        test_predictions = self.model.predict([r[1:] for r in test_data])

        for i in range(len(test_data)):
            if len(self.target_columns) == 1:
                yield (test_data[i][0],) + (test_predictions[i],)
            else:
                yield (test_data[i][0],) + tuple(test_predictions[i, :])


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
