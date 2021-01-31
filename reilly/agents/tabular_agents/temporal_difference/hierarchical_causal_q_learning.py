import numpy as np

from .hierarchical_td import HierarchicalTD
from .causal_td import CausalTD


class HierarchicalCausalQLearning(CausalTD, HierarchicalTD, object):

    def __repr__(self):
        return "HierarchicalCausalQLearning: " + "alpha=" + str(self._alpha) + \
            ", gamma=" + str(self._gamma) + \
            ", epsilon=" + str(self._epsilon) + \
            ", e-decay=" + str(self._e_decay)

    def reset(self, init_state:int, super_init_state:int, *args, **kwargs) -> None:
        self._super_agent.reset(init_state=super_init_state)
        self._S = init_state
        self._A = self._select_action(self._policy[init_state], state=init_state, env=kwargs['env'])
  
    
    def update(self, n_S: int, R: float, done: bool, *args, **kwargs) -> None:
        if kwargs['training']:
            self._Q[self._S, self._A] += self._alpha * \
                (R + (self._gamma * np.max(self._Q[n_S])) - self._Q[self._S, self._A])
            self._policy_update(self._S, self._policy, self._Q)

        # If the super agent reach current subgoal do action and update super agent
        if kwargs['env'].super_reach_current_subgoal():
            action = self._super_agent.get_action()
            next_state, reward, _, _ = kwargs['env'].super_run_step(action)
            self._super_agent.update(
                next_state,
                reward,
                done,
                training=True,
            )

        self._S = n_S
        self._A = self._select_action(self._policy[n_S], state=n_S, env=kwargs['env'], hierarchical=True)

        if done:
            self._epsilon *= self._e_decay
            self._epsilon = max(self._epsilon, self._min_epsilon)


