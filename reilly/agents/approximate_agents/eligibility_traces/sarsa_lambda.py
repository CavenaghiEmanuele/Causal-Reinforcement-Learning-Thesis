import numpy as np
from typing import List

from .eligibility_trace import EligibilityTrace


class SarsaLambda(EligibilityTrace, object):

    def __repr__(self):
        return "Sarsa Lambda: " + "alpha=" + str(self._alpha) + \
            ", gamma=" + str(self._gamma) + \
            ", epsilon=" + str(self._epsilon) + \
            ", lambda=" + str(self._lambda) + \
            ", e-decay=" + str(self._e_decay)


    def update(self, n_S: int, R: float, done: bool, *args, **kwargs) -> None:

        if done and kwargs['training']:
            self._Q_estimator.update(self._S, self._A, R)
            self._epsilon *= self._e_decay
            return

        n_A = self._select_action(n_S)

        if kwargs['training']:
            G = R + (self._gamma * self._Q_estimator.predict(n_S, n_A))
            self._Q_estimator.update(self._S, self._A, G)
            self._Q_estimator.update_traces(self._gamma, self._lambda)

        self._S = n_S
        self._A = n_A
