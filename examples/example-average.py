#!/usr/bin/env python3

import sys

import mltoys


class AverageModel(mltoys.types.MLToyBase):
    def __init__(self, columns, feature_columns, target_columns, loss_function):
        super().__init__(
            columns=columns,
            feature_columns=feature_columns,
            target_columns=target_columns,
            loss_function=loss_function,
        )

        self._training_rows = 0
        self._training_sums = [0.0 for c in self.target_columns]

    def fit(self, training_data):
        """
        Records distinct values from target columns.
        """

        num_target_columns = len(self.target_columns)
        for r in training_data:
            self._training_rows += 1
            for (i, v) in enumerate(r[-num_target_columns:]):
                self._training_sums[i] += v

        self._predictions = tuple(v / self._training_rows for v in self._training_sums)

    def predict(self, test_data):
        """
        Pick a random value recorded for each column.
        """

        for r in test_data:
            yield (r[0],) + self._predictions


def main():
    print("average model:")

    for factory in mltoys.get_factories():
        mltoys.test_models(model_class=AverageModel, factory=factory)

    return 0


############################################################
# startup handling #########################################
############################################################

if __name__ == "__main__":
    sys.exit(main())
