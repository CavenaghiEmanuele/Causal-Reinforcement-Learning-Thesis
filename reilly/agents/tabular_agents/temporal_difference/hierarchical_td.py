import numpy as np

from abc import ABC

from .temporal_difference import TemporalDifference
from .q_learning import QLearning


class HierarchicalTD(TemporalDifference, ABC, object):

    def __init__(
        self,
        states: int,
        actions: int,
        super_states: int,
        super_actions: int,
        alpha: float,
        epsilon: float,
        gamma: float,
        epsilon_decay: float = 1,
        min_epsilon: float = 0.05,
        *args,
        **kwargs
    ):
        super().__init__(states=states, actions=actions, alpha=alpha, epsilon=epsilon, gamma=gamma, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon)
        self._super_agent = QLearning(states=super_states, actions=super_actions, alpha=0.2, epsilon=0.05, gamma=0.95)

    def reset(self, init_state:int, super_init_state:int, *args, **kwargs) -> None:
        self._super_agent.reset(init_state=super_init_state)
        self._S = init_state
        self._A = self._select_action(self._policy[init_state])
