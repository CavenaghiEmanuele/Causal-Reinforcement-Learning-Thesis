import numpy as np
import networkx as nx
import matplotlib as plt
import random

from typing import Dict, List, Tuple

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD

from .base import Base

'''
States:
    S: corticosteroid level 
        - 0: low
        - 1: high
    M: patient's mood
        - 0: positive
        - 1: negative

Actions:
    - 0: no drug
    - 1: give drug

Rewards:
    - 0: not healthy
    - 1: healthy 
'''
class ConfounderInTime2(Base):

    _done: bool
    _state: List[int]
    _step: int
    _max_step: int
    _observe_confounder: bool
    _reward_probs: Dict
    _next_state_probs: Dict
    _next_M_probs: Dict

    def __init__(
        self, 
        build_causal_model:bool=False, 
        observe_confounder:bool=True, 
        max_steps:int=100, 
        reward_probs:Dict=None
        ):

        self._done = False
        self._observe_confounder = observe_confounder
        self._state = [1, 0]
        self._max_step = max_steps

        if reward_probs == None:
            # P(R=1) - format: S, Mt+1, Mt, X
            self._reward_probs = {
                '[0, 0, 0, 0]': 0.2,
                '[0, 0, 0, 1]': 0.9,
                '[0, 0, 1, 0]': 0.9,
                '[0, 0, 1, 1]': 0.2,
                '[0, 1, 0, 0]': 0.8,
                '[0, 1, 0, 1]': 0.3,
                '[0, 1, 1, 0]': 0.3,
                '[0, 1, 1, 1]': 0.8,

                '[1, 0, 0, 0]': 0.7,
                '[1, 0, 0, 1]': 0.2,
                '[1, 0, 1, 0]': 0.2,
                '[1, 0, 1, 1]': 0.7,
                '[1, 1, 0, 0]': 0.1,
                '[1, 1, 0, 1]': 0.8,
                '[1, 1, 1, 0]': 0.8,
                '[1, 1, 1, 1]': 0.1}
        else:
            self._reward_probs = reward_probs

        # P(S=1) - format: X, S
        self._next_state_probs = {
            '[0, 0]': 0.1,
            '[0, 1]': 0.7,
            '[1, 0]': 0.3,
            '[1, 1]': 0.2,
        }

        # P(M=1) - format: X, M 
        self._next_M_probs = {
            '[0, 0]': 0.7,
            '[0, 1]': 0.3,
            '[1, 0]': 0.2,
            '[1, 1]': 0.4,
        }

        if build_causal_model:
            self.build_causal_model()

        self.reset()
    
    @property
    def states(self) -> int:
        if self._observe_confounder:
            return 4
        else:
            return 2
    

    def encode(self, state:Tuple) -> int:
        return state[0] + state[1]*2
    
    def reset(self, *args, **kwargs) -> int:
        # S, M
        self._state = [1, random.randint(0,1)]
        self._step = 0
        self._done = False
        if self._observe_confounder:
            return self.encode(self._state)
        else:
            return self._state[0]

    def _run_step(self, action):

        Mt = self._state[1]
        # M = self._state[1]
        self._state[1] = np.random.binomial(size=1, n=1, p= self._next_M_probs[str([action, self._state[1]])])[0] # P(M=1)

        # Reward computation
        reward = np.random.binomial(size=1, n=1, p= self._reward_probs[str(self._state + [Mt, action])])[0] # P(R=1)
        
        # Next state compute 
        # S = self._state[0]
        self._state[0] = np.random.binomial(size=1, n=1, p= self._next_state_probs[str([action, self._state[0]])])[0] # P(S=1)


        # Done computation
        done = False
        self._step += 1
        if self._step > self._max_step:
            done = True
        
        if self._observe_confounder:
            return self.encode(self._state), reward, done, {} # info = {}
        else:
            return self._state[0], reward, done, {} # info = {}
    

    ###############################################
    # Causal section
    ###############################################
    
    def build_causal_model(self):

        self._causal_model = BayesianModel(
            [               
                ('Mt', 'Mt+1'),
                ('Mt', 'Xt+1'),
                ('Mt', 'Yt+1'),

                ('St+1', 'Xt+1'),
                ('St+1', 'Yt+1'),

                ('Xt+1', 'Yt+1'),
                ('Xt+1', 'Mt+1'),

                ('Mt+1', 'Yt+1')
            ])

        cpd_Mt = TabularCPD(
            variable='Mt',
            variable_card=2,
            values=[[0.5], [0.5]],
            state_names={'Mt':['positive', 'negative']})
        cpd_St_1 = TabularCPD(
            variable='St+1',
            variable_card=2,
            values=[[0.5], [0.5]],
            state_names={'St+1':['low', 'high']})
        cpd_Xt_1 = TabularCPD(
            variable='Xt+1',
            variable_card=2,
            values=[
                [0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5]
                ],
            evidence=['Mt', 'St+1'],
            evidence_card=[2,2],
            state_names={
                'Xt+1':['no drug', 'give drug'],
                'Mt':['positive', 'negative'],
                'St+1':['low', 'high']
                })
        cpd_Mt_1 = TabularCPD(
            variable='Mt+1',
            variable_card=2,
            values=[
                [0.3, 0.7, 0.8, 0.6],
                [0.7, 0.3, 0.2, 0.4]
                ],
            evidence=['Xt+1', 'Mt'],
            evidence_card=[2,2],
            state_names={
                'Mt+1':['positive', 'negative'],
                'Xt+1':['no drug', 'give drug'],
                'Mt':['positive', 'negative']
                })
        cpd_Yt_1 = TabularCPD(
            variable='Yt+1',
            variable_card=2,
            values=[
                [0.8, 0.1, 0.1, 0.8, 0.2, 0.7, 0.7, 0.2, 0.3, 0.8, 0.8, 0.3, 0.9, 0.2, 0.2, 0.9],
                [0.2, 0.9, 0.9, 0.2, 0.8, 0.3, 0.3, 0.8, 0.7, 0.2, 0.2, 0.7, 0.1, 0.8, 0.8, 0.1]
                ],
            evidence=['St+1', 'Mt+1', 'Mt', 'Xt+1'],
            evidence_card=[2, 2, 2, 2],
            state_names={
                'Yt+1':['not healthy', 'healthy'],
                'St+1':['low', 'high'],
                'Mt+1':['positive', 'negative'],
                'Mt':['positive', 'negative'],
                'Xt+1':['no drug', 'give drug']
                })

        self._causal_model.add_cpds(
            cpd_Mt, cpd_St_1, cpd_Xt_1, cpd_Mt_1, cpd_Yt_1)
        
        self._causal_model.check_model()

    def get_causal_model(self):
        return self._causal_model

    def get_target(self):
        return 'Yt+1'
    
    def get_good_target_value(self):
        return 'healthy'

    def get_evidence(self, state):
        return {
            'St+1': self._state[0],
            'Mt': self._state[1]
        }

    def get_action(self):
        return 'Xt+1'

    def get_action_values(self):
        return ['no drug', 'give drug']

    def plot_causal_model(self):
        nx.draw(self._causal_model, with_labels=True)
        plt.show()

    def causal_action_to_env_action(self, causal_action):
        if causal_action == 'no drug':
            return 0
        elif causal_action == 'give drug':
            return 1
    
    def get_confounder(self):
        return 'Mt+1'

    def get_confounder_values(self):
        return ['positive', 'negative']
    
    def causal_confounder_to_env_confounder(self, causal_confounder):
        if causal_confounder == 'positive':
            return 0
        elif causal_confounder == 'negative':
            return 1

    def get_agent_intent(self):
        return 0
