import numpy as np

from typing import Tuple

from .abstract_gym import GymEnvironment
from gym_maze.envs.maze_env import MazeEnv

class Maze(GymEnvironment):

    _current_state: Tuple
    _reached_subgoals: bool

    def __init__(self, build_causal_model:bool=False, confounders:bool=False, x_size:int=10, y_size:int=10, mode:str=None):

        self._env = MazeEnv(maze_size=(x_size, y_size), mode=mode, enable_render=True)
        self._subgoal = 0
        self._reached_middle_subgoal = False

        #if build_causal_model:
        #    self.build_causal_model()
        #self._confounders = confounders
        #self._not_a_passenger = False

        self.reset()

    @property
    def states(self) -> int:
        return int(np.prod(self._env.observation_space.high + np.ones([1, 1])))


    def run_step(self, action, hierarchical:bool=False, *args, **kwargs):
        action = self._convert_action(action) # Here actions are string ('N', 'S', 'E', 'W')
        next_state, reward, done, _ = self._env.step(action)
        agent_reward = reward
        self._current_state = next_state
        next_state = self._flat_state(next_state)

        info = {'wins': 0, 'steps': 1}
        if done and reward == 1:
            info['wins'] = 1

        # Hierarchical modification of next state and reward
        if hierarchical:
            if (self._subgoal == 1 or self._subgoal == 2) and not self._reached_middle_subgoal:
                self._reached_middle_subgoal = True
                agent_reward = 0.5
            next_state += self.states * self._subgoal

        return next_state, reward, agent_reward, done, info

    def reset(self, hierarchical:bool=False, *args, **kwargs) -> int:
        self.current_state = self._env.reset()
        if hierarchical:
            # World reset, hierarchical MDP reset
            return (0, self._super_reset()) # Agent always start from the top left corner (0, 0) --> 0
        return 0 # Agent always start from the top left corner (0, 0) --> 0
    
    def _convert_action(self, action: int) -> str:
        return ('N', 'S', 'E', 'W')[action]
    
    def _flat_state(self, state) -> int:
        rows = self._env.observation_space.high[0] + 1
        return state[0] * rows + state[1]


    ###############################################
    # Hierarchical section
    ###############################################
    @property
    def states_hierarchical(self) -> int:
        return int(np.prod(self._env.observation_space.high + np.ones([1, 1]))) * 4

    # +----+----+
    # | Q1 | Q2 |
    # +----+----+ 
    # | Q3 | Q4 |
    # +----+----+
    # State 0 (Q1): first quarter (start point is here) 
    # State 1 (Q2): subgoal reach second quarter
    # State 2 (Q3): subgoal reach third quarter
    # State 3 (Q4): global goal reach fourth quarter (end point is here)
    # Q1 <--> Q2 <--> Q4
    # Q1 <--> Q3 <--> Q4
    @property
    def super_states(self) -> int:
        return 4

    # Action 0: stay in the same state
    # Action 1: move to upper state
    # Action 2: move to lower state
    @property
    def super_actions(self) -> int:
        return 3

    def _super_reset(self, *args, **kwargs) -> int:
        self._subgoal = 0
        self._reached_middle_subgoal = False
        return 0 #Starting state with no subgoal

    # next_state, reward, done, info
    def super_run_step(self, action, cum_reward, *args, **kwargs):
        # action 0 stay in the same state = do nothing
        if action == 1:
            if self._subgoal == 0 or self._subgoal == 2:
                self._subgoal += 1
            elif self._subgoal == 1:
                self._subgoal += 2
            elif self._subgoal == 3:
                self._subgoal -= 2
        else: # action == 2
            if self._subgoal == 1 or self._subgoal == 3:
                self._subgoal -= 1
            elif self._subgoal == 0:
                self._subgoal += 2
            elif self._subgoal == 2:
                self._subgoal -= 2

        if self._subgoal == 3:
            reward = max(0, cum_reward)
            return (self._subgoal, reward + 1, None, None)
        return (self._subgoal, 0, None, None)

    def super_reach_current_subgoal(self, *args, **kwargs):
        # Env must be a square
        limit = int(self._env.observation_space.high[0] / 2)

        x, y = self._current_state
        if x < limit:
            if y < limit and self._subgoal == 0: # Q1
                return True
            elif y >= limit and self._subgoal == 2: # Q3  
                return True
        else: # x >= limit
            if y < limit and self._subgoal == 1: # Q2
                return True
            elif y >= limit and self._subgoal == 3: #Q4 
                return True
        return False

    def get_current_subgoal(self):
        return self._subgoal
