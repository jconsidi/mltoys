# mltoys/types.py

from typing import Tuple


class MLToyBase:
    """
    Base class defining shared conventions.
    """

    def __init__(self, columns, feature_columns, target_columns):
        self._columns = tuple(columns)
        self._feature_columns = tuple(feature_columns)
        self._target_columns = tuple(target_columns)

        if self.columns != ("id",) + self.feature_columns + self.target_columns:
            print(self.columns)
            print(self.feature_columns)
            print(self.target_columns)
            print(("id",) + self.feature_columns + self.target_columns)
            raise RuntimeError(
                "column order must be id, feature columns, target columns"
            )

    @property
    def columns(self) -> Tuple[str]:
        return self._columns

    @property
    def feature_columns(self) -> Tuple[str]:
        return self._feature_columns

    @property
    def target_columns(self) -> Tuple[str]:
        return self._target_columns


class MLToyInstance(MLToyBase):
    def __init__(
        self, factory, seed, columns, feature_columns, target_columns, loss_function
    ):
        super().__init__(
            columns=columns,
            feature_columns=feature_columns,
            target_columns=target_columns,
        )

        self._factory = factory
        self._seed = seed
        self._loss_function = loss_function

    def calculate_loss(self, test_predictions) -> float:
        raise NotImplementedError()

    @property
    def factory(self):
        return self._factory

    @property
    def loss_function(self) -> str:
        return self._loss_function

    def seed(self):
        return self._seed

    @property
    def training_data(self):
        raise NotImplementedError()

    @property
    def test_data(self):
        raise NotImplementedError()


class MLToyFactory:
    def get_instance(self, seed=None):
        raise NotImplementedError()
