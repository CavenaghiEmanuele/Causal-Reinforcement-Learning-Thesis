import sys
import gym
import gym_maze
import numpy as np
import matplotlib.pyplot as plt

from .abstract_gym import GymEnvironment
from gym_maze.envs.maze_env import MazeEnv

class Maze(GymEnvironment):

    def __init__(self, build_causal_model:bool=False, confounders:bool=False, x_size:int=10, y_size:int=10, mode:str=None):
        
        self._env = MazeEnv(maze_size=(x_size, y_size), mode=mode, enable_render=True)
        self._done = False
        #self._subgoal = 0
        #self._passenger_on_taxi = False
        #if build_causal_model:
        #    self.build_causal_model()
        #self._confounders = confounders
        #self._not_a_passenger = False

        self.reset()

    @property
    def states(self) -> int:
        return int(np.prod(self._env.observation_space.high + np.ones([1, 1])))


    def run_step(self, action, *args, **kwargs):
        action = self._convert_action(action) # Here actions are string ('N', 'S', 'E', 'W')
        next_state, reward, done, _ = self._env.step(action)
        next_state = self._flat_state(next_state)

        info = {'wins': 0, 'steps': 1}
        if done and reward == 1:
            info['wins'] = 1
        self._done == done

        return next_state, reward, done, info

    def reset(self, hierarchical:bool=False, *args, **kwargs) -> int:
        self._env.reset()
        return 0 # Agent always start from the top left corner (0, 0) --> 0
    
    def _convert_action(self, action: int) -> str:
        return ('N', 'S', 'E', 'W')[action]
    
    def _flat_state(self, state) -> int:
        rows = self._env.observation_space.high[0] + 1
        return state[0] * rows + state[1]

