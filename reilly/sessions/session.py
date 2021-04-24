from typing import Dict, List
from tqdm import trange

import pandas as pd
import time

from ..agents import Agent
from ..environments import Environment


class Session(object):

    __slots__ = ['_env', '_agent', '_label', '_max_steps']

    _env: Environment
    _agent: Agent
    _label: str
    _max_steps: int

    def __init__(self, env: Environment, agent: Agent, max_steps: int=200, *args, **kwargs):
        self._env = env
        self._agent = agent
        self._label = "ID: {}, Params: {}".format(id(agent), agent)
        self._max_steps = max_steps

    def run(self, episodes: int, test_offset: int, test_samples: int, render: bool = False, *args, **kwargs) -> pd.DataFrame:
        out = []
        self._reset_env()
        for episode in trange(episodes, position=kwargs.get('position', 0)):
            self._run_train()
            if (episode + 1) % test_offset == 0:
                out.append(
                    self._run_test(
                        episode // test_offset,
                        test_samples, 
                        render
                    )
                )
        return pd.concat(out)

    def _run_train(self) -> None:
        step = 0
        done = False
        while not done:
            action = self._agent.get_action()
            
            next_state, reward, done, _ = self._env.run_step(
                action,
                id=id(self._agent),
                mode='test',
                t=step
            )
                
            self._agent.update(
                next_state,
                reward,
                done,
                training=True,
                t=step,
                env=self._env,
            )
            if step >= self._max_steps:
                self._reset_env()
                return
            step += 1
        self._reset_env()

    def _run_test(self, test: int, test_samples: int, render: bool = False) -> pd.DataFrame:
        self._reset_env()
        out = []
        for sample in range(test_samples):
            step = 0
            done = False
            while not done:
                if render:
                    self._env.render()
                    time.sleep(0.1)
                action = self._agent.get_action()
                
                next_state, reward, done, info = self._env.run_step(
                    action,
                    id=id(self._agent),
                    mode='test',
                    t=step
                )

                self._agent.update(
                    next_state,
                    reward,
                    done,
                    training=False,
                    t=step,
                    env=self._env
                )
                out.append({
                    'test': test,
                    'sample': sample,
                    'step': step,
                    'agent': self._label,
                    'reward': reward,
                    **info
                })
                if step >= self._max_steps:
                    self._reset_env()
                    return pd.DataFrame(out)
                step += 1
            self._reset_env()
        return pd.DataFrame(out)

    def _reset_env(self) -> None:
        init_state = self._env.reset(id=id(self._agent))
        self._agent.reset(init_state, env=self._env)
