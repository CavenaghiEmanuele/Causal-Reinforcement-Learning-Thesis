import numpy as np
import operator
import random

from abc import ABC

from .temporal_difference import TemporalDifference
from ....utils.causal_inference import causal_query


class CausalTD(TemporalDifference, ABC, object):

    _cache_inference = {}

    def __init__(
        self,
        states: int,
        actions: int,
        alpha: float,
        epsilon: float,
        gamma: float,
        epsilon_decay: float = 1,
        min_epsilon: float = 0.05,
        causal_threshold: float = 0.0,
        *args,
        **kwargs
    ):

        self._causal_threshold = causal_threshold
        super().__init__(
            states=states,
            actions=actions,
            alpha=alpha,
            epsilon=epsilon,
            gamma=gamma,
            epsilon_decay=epsilon_decay,
            min_epsilon=min_epsilon
            )

    def reset(self, init_state:int, *args, **kwargs) -> None:
        self._S = init_state
        self._A = self._select_action(self._policy[init_state], state=init_state, env=kwargs['env'])

    def _select_action(self, policy_state, state, env):

        try:
            a = self._cache_inference[(state)]
        except:
            a = self._inferenced_selection(env=env, state=state)
            self._cache_inference.update({(state) : a})
        
        if a == None or random.uniform(0, 1) < self._epsilon:
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
        if candidate[1] > self._causal_threshold:
            return env.causal_action_to_env_action(candidate[0])
        else:
            return None
