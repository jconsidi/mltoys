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
    print("demo model using training average for all predictions:")

    for factory in mltoys.get_factories():
        test_losses = []
        for instance in factory.get_instances(count=10):
            model = AverageModel(
                columns=instance.columns,
                feature_columns=instance.feature_columns,
                target_columns=instance.target_columns,
                loss_function=instance.loss_function,
            )

            model.fit(instance.training_data)

            test_loss = instance.calculate_loss(model.predict(instance.test_data))
            test_losses.append(test_loss)

        print(
            f"{factory.__class__.__name__} : min/mean/max = {min(test_losses):.4f}/{sum(test_losses)/len(test_losses):.4f}/{max(test_losses):.4f}"
        )

    return 0


############################################################
# startup handling #########################################
############################################################

if __name__ == "__main__":
    sys.exit(main())
