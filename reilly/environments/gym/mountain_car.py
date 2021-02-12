import gym
import numpy as np

from .abstract_gym import GymEnvironment

"""
Description:
    The agent (a car) is started at the bottom of a valley. For any given
    state the agent may choose to accelerate to the left, right or cease
    any acceleration.

Observation:
    Type: [d, d] # d = digit
    Num    Observation               Min            Max
    0      Car Position              -1.2           0.6
    1      Car Velocity              -0.07          0.07

Actions:
    Type: Discrete(3)
    Num    Action
    0      Accelerate to the Left
    1      Don't accelerate
    2      Accelerate to the Right
    Note: This does not affect the amount of velocity affected by the
    gravitational pull acting on the car.

Reward:
        Reward of 0 is awarded if the agent reached the flag (position = 0.5)
        on top of the mountain.
        Reward of -1 is awarded if the position of the agent is less than 0.5.

Starting State:
        The position of the car is assigned a uniform random value in
        [-0.6 , -0.4].
        The starting velocity of the car is always assigned to 0.
        
Episode Termination:
        The car position is more than 0.5
        Episode length is greater than 200
"""
class MountainCar(GymEnvironment):

    _done: bool
    _subgoal: int
    _reach_left: bool

    def __init__(self):
        self._env = gym.make("MountainCar-v0")
        self._reach_left = False
        self._done = False
        self._subgoal = 0

        self.reset()

    @property
    def states(self):
        '''
        Discretize the state space for tabular agents. One simple way in which
        this can be done is to round the first element of the state vector to
        the nearest 0.1 and the second element to the nearest 0.01, and then
        (for convenience) multiply the first element by 10 and the second by 100.
        '''
        return 285

    def run_step(self, action, hierarchical:bool=False, *args, **kwargs):
        next_state, reward, done, _ = self._env.step(action)
        agent_reward = reward
        raw_next_state = next_state
        next_state = self._flat_state(next_state)

        info = {'steps': 1}
        self._done == done

        # Hierarchical modification of next state and reward
        if hierarchical:
            if self._subgoal == 1 and (not self._reach_left) and raw_next_state[0] <= -1.1:
                agent_reward = 5
                self._reach_left = True
            next_state += self.states * self._subgoal

        return next_state, reward, agent_reward, done, info

    def reset(self, hierarchical:bool=False, *args, **kwargs) -> int:
        if hierarchical:
            # World reset, hierarchical MDP reset
            return (self._flat_state(self._env.reset()), self._super_reset())
        return self._flat_state(self._env.reset())

    def _flat_state(self, state) -> int:
        state_adj = (state - self._env.observation_space.low)*np.array([10, 100])
        state_adj = np.round(state_adj).astype(int)
        return state_adj[0] + 19*state_adj[1]

    
    ###############################################
    # Hierarchical section
    ###############################################

    @property
    def states_hierarchical(self) -> int:
        return self.states * 3

    # State 0: no subgoal
    # State 1: subgoal reach left size of mountain (position = -1.0)
    # State 2: subgoal reach global goal
    # 0 <--> 1 <--> 2
    @property
    def super_states(self) -> int:
        return 3

    # Action 0: stay in the same state
    # Action 1: move in left state
    # Action 2: move in right state
    @property
    def super_actions(self) -> int:
        return 3

    def _super_reset(self, *args, **kwargs) -> int:
        self._subgoal = 0
        self._reach_left = False
        self._done = False
        return 0 #Starting state with no subgoal

    # next_state, reward, done, info
    def super_run_step(self, action, cum_reward, *args, **kwargs):
        # action 0 stay in the same state = do nothing
        if action == 1:
            self._subgoal -= 1
            if self._subgoal < 0:
                self._subgoal = 0
        elif action == 2:
            self._subgoal += 1
            if self._subgoal >= 2:
                self._subgoal = 2

        if self._subgoal == 2:
            reward = max(0, cum_reward)
            return (self._subgoal, reward + 1, None, None)
        return (self._subgoal, 0, None, None)
        

    def super_reach_current_subgoal(self, *args, **kwargs):
        if self._subgoal == 0:
            return True
        elif self._subgoal == 1 and self._reach_left:
            return True
        elif self._subgoal == 2:
            return True
        return False

    def get_current_subgoal(self):
        return self._subgoal


