import numpy as np
import operator

from .causal_td import CausalTD
from ....utils.causal_inference import causal_query


class CausalQLearning(CausalTD, object):

    def __repr__(self):
        return "CausalQLearning: " + "alpha=" + str(self._alpha) + \
            ", gamma=" + str(self._gamma) + \
            ", epsilon=" + str(self._epsilon) + \
            ", e-decay=" + str(self._e_decay)

    def update(self, n_S: int, R: float, done: bool, *args, **kwargs) -> None:
        if kwargs['training'] and not kwargs['causal_action']:
            self._Q[self._S, self._A] += self._alpha * \
                (R + (self._gamma * np.max(self._Q[n_S])) - self._Q[self._S, self._A])
            self._policy_update(self._S, self._policy, self._Q)

        self._S = n_S
        self._A = self._select_action(self._policy[n_S], state=n_S, env=kwargs['env'])

        if done:
            self._epsilon *= self._e_decay
            self._epsilon = max(self._epsilon, self._min_epsilon)
