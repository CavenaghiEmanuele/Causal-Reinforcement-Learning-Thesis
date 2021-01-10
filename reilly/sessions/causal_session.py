from typing import Dict, List
from tqdm import trange

import pandas as pd

from .session import Session
from ..agents import Agent
from ..environments import Environment


class CausalSession(Session):

    def __init__(self, env: Environment, agent: Agent, *args, **kwargs):
        self._env = env
        self._agent = agent
        self._label = "ID: {}, Params: {}".format(id(agent), agent)

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
                causal_model=self._env.get_causal_model()
            )
            step += 1
        self._reset_env()

    def _run_test(self, test: int, test_samples: int, render: bool = False) -> pd.DataFrame:
        self._reset_env()
        if render:
            self._env.render()
        out = []
        for sample in range(test_samples):
            step = 0
            done = False
            while not done:
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
                    causal_model=self._env.get_causal_model()
                )
                out.append({
                    'test': test,
                    'sample': sample,
                    'step': step,
                    'agent': self._label,
                    'reward': reward,
                    **info
                })
                step += 1
            self._reset_env()
        return pd.DataFrame(out)
    
    def _reset_env(self) -> None:
        init_state = self._env.reset(id=id(self._agent))
        self._agent.reset(init_state, causal_model=self._env.get_causal_model())
