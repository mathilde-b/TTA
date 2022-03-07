from typing import Any, Callable, List, Tuple
from operator import add

from utils import map_, uc_


class DummyScheduler(object):
    def __call__(self, epoch: int, optimizer: Any, loss_fns: List[Callable], loss_weights: List[float]) \
            -> Tuple[float, List[Callable], List[float]]:
        return optimizer, loss_fns, loss_weights


class AddWeightLoss():
    def __init__(self, to_add: List[float]):
        self.to_add: List[float] = to_add

    def __call__(self, epoch: int, optimizer: Any, loss_fns: List[Callable], loss_weights: List[float]) \
            -> Tuple[float, List[Callable], List[float]]:
        assert len(self.to_add) == len(loss_weights)
        new_weights: List[float] = map_(uc_(add), zip(loss_weights, self.to_add))

        print(f"Loss weights went from {loss_weights} to {new_weights}")

        return optimizer, loss_fns, new_weights


class StealWeight():
    def __init__(self, to_steal: float):
        self.to_steal: float = to_steal

    def __call__(self, epoch: int, optimizer: Any, loss_fns: List[Callable], loss_weights: List[float]) \
            -> Tuple[float, List[Callable], List[float]]:
        a, b = loss_weights
        new_weights: List[float] = [max(0.1, a - self.to_steal), b + self.to_steal]

        print(f"Loss weights went from {loss_weights} to {new_weights}")

        return optimizer, loss_fns, new_weights