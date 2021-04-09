from networkx.generators.triads import TRIAD_EDGES
import numpy as np
import operator

from abc import ABC

from .temporal_difference import TemporalDifference
from ....utils.causal_inference import causal_query


class CausalTD(TemporalDifference, ABC, object):

    _cache_inference = {}

    def reset(self, init_state:int, *args, **kwargs) -> None:
        self._S = init_state
        self._A = self._select_action(self._policy[init_state], state=init_state, env=kwargs['env'])

    def _select_action(self, policy_state, state, env):
        intent = env.get_agent_intent()
        try:
            a = self._cache_inference[(state, intent)]
        except:
            a = self._inferenced_selection(env=env, state=state)
            self._cache_inference.update({(state, intent) : a})

        if a == None:
            return np.random.choice(range(self._actions), p=policy_state)
        return a

    def _inferenced_selection(self, env, state):
        target = env.get_target()
        actions = env.get_actions()
        query = causal_query(
            target=target,
            evidence=env.get_evidence(state),
            actions={action:env.get_action_values(action) for action in actions},
            model=env.get_causal_model()
            )

        # Get the value for each possible action as dict 
        values = {
            value : query[action].get_value(**{target:env.get_good_target_value(), action:value}) 
            for action in actions
            for value in env.get_action_values(action)
            }

        # Select the action with the highest MAP
        candidate = max(values.items(), key=operator.itemgetter(1))

        return env.causal_action_to_env_action(candidate[0])

