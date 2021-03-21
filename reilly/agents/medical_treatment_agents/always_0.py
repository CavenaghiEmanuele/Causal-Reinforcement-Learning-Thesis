import numpy as np

from ..agent import Agent


class Always0(Agent, object):

    _A: int
    
    def __repr__(self):
        return "Always 0 agent"

    def _select_action(self) -> int:
        return 0

    def update(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        self._A = self._select_action()
