import numpy as np

from .agent import Agent


class Random(Agent, object):

    _A: int

    def __init__(self, actions: int):
        self._actions = actions
    
    def __repr__(self):
        return "Random agent"

    def _select_action(self) -> int:
        return np.random.choice(range(self._actions))

    def update(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        self._A = self._select_action()
