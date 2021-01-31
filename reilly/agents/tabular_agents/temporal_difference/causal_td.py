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

    def _select_action(self, policy_state, state, env, hierarchical:bool=False):
        try:
            a = self._cache_inference[state]
        except:
            a = self._inferenced_selection(env=env, state=state, hierarchical=hierarchical)
            self._cache_inference.update({state : a})

        if a == None:
            return np.random.choice(range(self._actions), p=policy_state)
        return a

    def _inferenced_selection(self, env, state, hierarchical:bool=False):
        target = env.get_target(hierarchical=hierarchical)
        actions = env.get_actions(hierarchical=hierarchical)
        query = causal_query(
            target=target,
            evidence=env.get_evidence(state, hierarchical=hierarchical),
            actions=actions,
            model=env.get_causal_model()
            )

        # For each action get the MAP value as dict {action1: (value:prob), action2: (value:prob), ...}
        map = {}
        for action in actions:
            map.update({action : max(
                {   
                    value : query[action].get_value(**{target:0, action:value}) # target=0 --searching for-->  target=True (0=True, 1=False)
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
