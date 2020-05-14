#!/usr/bin/env python3

import sys

from sklearn.linear_model import LinearRegression

import mltoys


class LinearModel(mltoys.types.MLToyBase):
    def __init__(self, columns, feature_columns, target_columns, loss_function):
        super().__init__(
            columns=columns,
            feature_columns=feature_columns,
            target_columns=target_columns,
            loss_function=loss_function,
        )

        self.model = LinearRegression(
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
        Predict value for each target column.
        """

        test_data = tuple(map(tuple, test_data))

        test_predictions = self.model.predict([r[1:] for r in test_data])
        if test_predictions.ndim == 1:
            test_predictions = test_predictions.reshape(shape=(len(test_data), 1))

        for i in range(len(test_data)):
            yield (test_data[i][0],) + tuple(test_predictions[i, :].tolist())


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
