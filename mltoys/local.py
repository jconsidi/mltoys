# mltoys/local.py

from .loss_functions import check_loss_function
from .loss_functions import get_loss_function
from .types import MLToyFactory
from .types import MLToyInstance


class MLToyInstanceLocal(MLToyInstance):
    def __init__(
        self,
        factory,
        seed,
        columns,
        feature_columns,
        target_columns,
        loss_function,
        training_data,
        test_data,
    ):
        if not check_loss_function(loss_function):
            raise RuntimeError("unrecognized loss function {!r}".format(loss_function))

        super().__init__(
            factory=factory,
            seed=seed,
            columns=columns,
            feature_columns=feature_columns,
            target_columns=target_columns,
            loss_function=loss_function,
        )

        # freeze training and test data
        self._training_data = tuple(map(tuple, training_data))
        self._test_data = tuple(map(tuple, test_data))

    def calculate_loss(self, test_outputs) -> float:
        test_outputs = list(test_outputs)
        if len(test_outputs) != len(self._test_data):
            raise RuntimeError("number of test scores does not match test data")

        num_target_columns = len(self.target_columns)
        test_expected = {r[0]: r[-num_target_columns:] for r in self._test_data}
        test_actual = {r[0]: r[-num_target_columns:] for r in test_outputs}

        if set(test_expected.keys()) != set(test_actual.keys()):
            raise RuntimeError("test id mismatch")

        loss_function = get_loss_function(self._loss_function)
        loss_total = sum(
            loss_function(test_expected[test_id], test_actual[test_id])
            for test_id in test_expected.keys()
        )
        loss_average = loss_total / len(self._test_data)

        return loss_average

    @property
    def training_data(self):
        return self._training_data

    @property
    def test_data(self):
        # only return row id and feature columns
        num_test_columns = 1 + len(self.feature_columns)
        return [r[:num_test_columns] for r in self._test_data]


class MLToyFactoryLocal(MLToyFactory):
    def get_instance(self, seed=None):
        raise NotImplementedError()

    def get_instances(self, count=10):
        for _ in range(count):
            yield self.get_instance()
