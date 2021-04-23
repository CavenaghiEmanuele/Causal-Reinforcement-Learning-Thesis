import numpy as np
import operator

from abc import ABC

from .temporal_difference import TemporalDifference
from ....utils.causal_inference import causal_query, confounder_query


class CausalTD(TemporalDifference, ABC, object):

    _cache_inference = {}
    _confounder_cache = {}

    def reset(self, init_state:int, *args, **kwargs) -> None:
        self._S = init_state
        self._A = self._select_action(self._policy[init_state], state=init_state, env=kwargs['env'])

    def _select_action(self, policy_state, state, env):

        intent = np.random.choice(range(self._actions), p=policy_state)
        try:
            confounder_value = self._confounder_cache[(state, intent)]
        except:
            confounder_value = self._predict_confounder(
                env=env,
                state=state,
                intent=intent
                )
            self._confounder_cache.update({(state, intent) : confounder_value})

        try:
            a = self._cache_inference[(state, confounder_value)]
        except:
            a = self._inferenced_selection(env=env, state=state)
            self._cache_inference.update({(state, confounder_value) : a})

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
    
    def _predict_confounder(self, env, state, intent):
        confounder = env.get_confounder()
        if confounder != None:
            query = confounder_query(
                confounder=confounder, 
                evidence=env.get_evidence(state),
                model=env.get_causal_model(),
                action= {env.get_action(): intent}
                )
            values = {
                value : query.get_value(**{confounder:value}) 
                for value in env.get_confounder_values()
            }
            # Select the action with the highest MAP
            prediction = max(values.items(), key=operator.itemgetter(1))
            return env.causal_confounder_to_env_confounder(prediction[0])
        return 0
