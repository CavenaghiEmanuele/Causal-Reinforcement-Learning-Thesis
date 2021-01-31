import numpy as np
import operator

from .temporal_difference import TemporalDifference
from ....utils.causal_inference import causal_query


class CausalQLearning(TemporalDifference, object):

    _cache_inference = {}

    def __repr__(self):
        return "CausalQLearning: " + "alpha=" + str(self._alpha) + \
            ", gamma=" + str(self._gamma) + \
            ", epsilon=" + str(self._epsilon) + \
            ", e-decay=" + str(self._e_decay)

    def reset(self, init_state:int, *args, **kwargs) -> None:
        self._S = init_state
        self._A = self._select_action(self._policy[init_state], state=init_state, env=kwargs['env'])

    def update(self, n_S: int, R: float, done: bool, *args, **kwargs) -> None:
        if kwargs['training']:
            self._Q[self._S, self._A] += self._alpha * \
                (R + (self._gamma * np.max(self._Q[n_S])) - self._Q[self._S, self._A])
            self._policy_update(self._S, self._policy, self._Q)

        self._S = n_S
        self._A = self._select_action(self._policy[n_S], state=n_S, env=kwargs['env'])

        if done: 
            self._epsilon *= self._e_decay
            self._epsilon = max(self._epsilon, self._min_epsilon)

    def _select_action(self, policy_state, state, env):
        try:
            a = self._cache_inference[state]
        except:
            a = self._inferenced_selection(env=env, state=state)
            self._cache_inference.update({state : a})

        if a == None:
            return np.random.choice(range(self._actions), p=policy_state)
        return a

    def _inferenced_selection(self, env, state):
        query = causal_query(
            target=env.get_target(),
            evidence=env.get_evidence(state),
            actions=env.get_actions(),
            model=env.get_causal_model()
            )

        # For each action get the MAP value as dict {action1: (value:prob), action2: (value:prob), ...}
        map = {}
        for action in env.get_actions():
            map.update({action : max(
                {
                    value : query[action].get_value(G=0, **{action:value}) # G=0 --searching for-->  G=True (0=True, 1=False)
                    for value in env.get_action_values(action)
                }.items(),
                key=operator.itemgetter(1))
            })

        # Select the action with the highest MAP
        candidate = max(map.items(), key=operator.itemgetter(1))

        # Check if the candidate action probability is above a certain threshold
        # and if it's better to do an action then do not
        if candidate[1][1] > 0.4 and candidate[1][0] == 1:
            return env.causal_action_to_env_action(candidate[0])
        return None
