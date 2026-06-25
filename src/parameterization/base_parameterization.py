"""Define the common parameterization interface."""

from abc import ABC, abstractmethod


class BaseParameterization(ABC):
    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def predict(self, x, F=None):
        pass
