import numpy as np
import operator

from abc import ABC

from .temporal_difference import TemporalDifference
from ....utils.causal_inference import causal_query


class CausalTD(TemporalDifference, ABC, object):

    _cache_inference = {}
    _confounder_cache = {}

    def reset(self, init_state:int, *args, **kwargs) -> None:
        self._S = init_state
        self._A = self._select_action(self._policy[init_state], state=init_state, env=kwargs['env'])

    def _select_action(self, policy_state, state, env):

        try:
            a = self._cache_inference[(state)]
        except:
            a = self._inferenced_selection(env=env, state=state)
            self._cache_inference.update({(state) : a})

        if a == None:
            return np.random.choice(range(self._actions), p=policy_state)
        return a

    def _inferenced_selection(self, env, state):
        target = env.get_target()
        action = env.get_action()
        query = causal_query(
            target=target,
            evidence=env.get_evidence(state),
            actions={action:env.get_action_values()},
            model=env.get_causal_model()
            )

        # Get the value for each possible action as dict 
        values = {
            value : query[env.get_action()].get_value(**{target:env.get_good_target_value(), env.get_action():value}) 
            for value in env.get_action_values()
            }

        # Select the action with the highest MAP
        candidate = max(values.items(), key=operator.itemgetter(1))
        return env.causal_action_to_env_action(candidate[0])
