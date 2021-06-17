import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from abc import ABC

from .agent import Agent
from ..structures import ActionValue, Policy


class TabularAgent(Agent, ABC, object):

    __slots__ = ['_Q', '_policy']

    def __init__(
        self,
        states: int,
        actions: int,
        alpha: float,
        epsilon: float,
        gamma: float,
        epsilon_decay: float = 1,
        min_epsilon: float = 0.05,
        *args,
        **kwargs
    ):
        self._Q = ActionValue(states, actions)
        self._policy = Policy(states, actions)
        self._alpha = alpha
        self._epsilon = epsilon
        self._gamma = gamma
        self._e_decay = epsilon_decay
        self._actions = actions
        self._min_epsilon = min_epsilon

    def _select_action(self, policy_state) -> int:
        return np.random.choice(range(self._actions), p=policy_state)

    def _policy_update(self, state: int, policy: Policy, Q: ActionValue) -> None:
        # Avoid choosing always the first move in case policy has the same value
        indices = [i for i, x in enumerate(Q[state]) if x == max(Q[state])]
        A_star = np.random.choice(indices)

        for A in range(self._actions):
            if A == A_star:
                policy[state, A] = 1 - self._epsilon + \
                    (self._epsilon / self._actions)
            else:
                policy[state, A] = self._epsilon / self._actions

    def plot_Q_values(self):
        _ = sns.heatmap(self._Q._action_value)
        plt.show()
