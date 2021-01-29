import numpy as np

from .temporal_difference import TemporalDifference
from .q_learning import QLearning


class HierarchicalQLearning(TemporalDifference, object):

    def __init__(
        self,
        states: int,
        actions: int,
        super_states: int,
        super_actions: int,
        alpha: float,
        epsilon: float,
        gamma: float,
        epsilon_decay: float = 1,
        min_epsilon: float = 0.05,
        *args,
        **kwargs
    ):
        super().__init__(states=states, actions=actions, alpha=alpha, epsilon=epsilon, gamma=gamma, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon)
        self._super_agent = QLearning(states=super_states, actions=super_actions, alpha=0.2, epsilon=0.05, gamma=0.95)

    def __repr__(self):
        return "Hierarchical QLearning: " + "alpha=" + str(self._alpha) + \
            ", gamma=" + str(self._gamma) + \
            ", epsilon=" + str(self._epsilon) + \
            ", e-decay=" + str(self._e_decay)

    def reset(self, init_state:int, super_init_state:int, *args, **kwargs) -> None:
        self._super_agent.reset(init_state=super_init_state)
        self._S = init_state
        self._A = self._select_action(self._policy[init_state])

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
        self._A = self._select_action(self._policy[n_S])

        if done:
            self._epsilon *= self._e_decay
            self._epsilon = max(self._epsilon, self._min_epsilon)
