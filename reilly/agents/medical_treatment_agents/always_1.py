import numpy as np

from ..agent import Agent


class Always1(Agent, object):

    _A: int
    
    def __repr__(self):
        return "Always 1 agent"

    def _select_action(self) -> int:
        return 1

    def update(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        self._A = self._select_action()
