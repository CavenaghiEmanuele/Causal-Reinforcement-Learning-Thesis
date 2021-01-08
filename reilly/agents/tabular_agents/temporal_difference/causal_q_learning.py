import numpy as np

from .temporal_difference import TemporalDifference


class CausalQLearning(TemporalDifference, object):

    def __repr__(self):
        return "CausalQLearning: " + "alpha=" + str(self._alpha) + \
            ", gamma=" + str(self._gamma) + \
            ", epsilon=" + str(self._epsilon) + \
            ", e-decay=" + str(self._e_decay)

    def update(self, n_S: int, R: float, done: bool, *args, **kwargs) -> None:
        if kwargs['training']:
            self._Q[self._S, self._A] += self._alpha * \
                (R + (self._gamma * np.max(self._Q[n_S])) - self._Q[self._S, self._A])
            self._policy_update(self._S, self._policy, self._Q)

        self._S = n_S
        self._A = self._select_action(self._policy[n_S])

        if done: 
            self._epsilon *= self._e_decay

    def _select_action(self, policy_state):
        a = self._inferenced_selection()
        if a == None:
            return np.random.choice(range(self._actions), p=policy_state)

    def _inferenced_selection(self):
        return None
