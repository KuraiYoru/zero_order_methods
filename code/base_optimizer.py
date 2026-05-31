from abc import ABC, abstractmethod
import numpy as np


class BaseOptimizer(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def step(self, obs) -> np.ndarray:
        """
        Accepts an Observation object and returns the next point (coordinates) for the query.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the internal state of the algorithm between runs.
        """
        pass
