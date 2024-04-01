from typing import Protocol


class Metric(Protocol):
    """Base class for all metrics."""

    def reset(self):
        pass

    def add(self, predicted, target):
        pass

    def value(self):
        pass
