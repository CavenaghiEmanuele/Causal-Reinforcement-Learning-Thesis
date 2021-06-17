import random
from typing import Dict

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel

from .abstract_causal_medical_treatment import AbstractCausalMedicalTreatment
from .abstract_medical_treatment import AbstractMedicalTreatment

'''
States:
    S: corticosteroid level 
        - 0: low
        - 1: high
    M: patient's mood
        - 0: positive
        - 1: negative
    E: socioeconomic status
        - 0: wealthy
        - 1: poor

Actions:
    - 0: no drug
    - 1: give drug

Rewards:
    - 0: not healthy
    - 1: healthy 
'''
class ConfounderInTime(AbstractMedicalTreatment, AbstractCausalMedicalTreatment):

    _next_state_probs: Dict
    _next_M_probs: Dict
    _next_E_probs: Dict

    def __init__(self, build_causal_model:bool=False, observe_confounder:bool=True, max_steps:int=100):

        reward_probs = {
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
            '[1, 1, 1, 1]': 0.1,
            }

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
            '[0, 1]': 0.2,
            '[1, 0]': 0.3,
            '[1, 1]': 0.2,
        }

        # P(E=1) - format: X, E
        self._next_E_probs = {
            '[0, 0]': 0.3,
            '[0, 1]': 0.2,
            '[1, 0]': 0.7,
            '[1, 1]': 0.2
        }

        if build_causal_model:
            self.build_causal_model()

        super().__init__(
            build_causal_model=build_causal_model, 
            observe_confounder=observe_confounder, 
            max_steps=max_steps, 
            reward_probs=reward_probs)
    
    def reset(self, *args, **kwargs) -> int:
        # S, M, E
        self._state = [1, random.randint(0,1), random.randint(0,1)]
        self._step = 0
        self._done = False
        if self._observe_confounder:
            return self.encode(self._state)
        else:
            return self._state[0]

    def _run_step(self, action):

        # M = self._state[1]
        self._state[1] = np.random.binomial(size=1, n=1, p= self._next_M_probs[str([action, self._state[1]])])[0] # P(M=1)
        # E = self._state[2]
        self._state[2] = np.random.binomial(size=1, n=1, p= self._next_E_probs[str([action, self._state[2]])])[0] # P(E=1)

        # Reward computation
        reward = np.random.binomial(size=1, n=1, p= self._reward_probs[str(self._state + [action])])[0] # P(R=1)
        
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

                ('Et', 'Et+1'),
                ('Et', 'Xt+1'),

                ('St+1', 'Xt+1'),
                ('St+1', 'Yt+1'),

                ('Xt+1', 'Yt+1'),
                ('Xt+1', 'Mt+1'),
                ('Xt+1', 'Et+1'),

                ('Et+1', 'Yt+1'),
                ('Mt+1', 'Yt+1')
            ])

        cpd_Mt = TabularCPD(
            variable='Mt',
            variable_card=2,
            values=[[0.5], [0.5]],
            state_names={'Mt':['positive', 'negative']})
        cpd_Et = TabularCPD(
            variable='Et',
            variable_card=2,
            values=[[0.5], [0.5]],
            state_names={'Et':['wealthy', 'poor']})
        cpd_St_1 = TabularCPD(
            variable='St+1',
            variable_card=2,
            values=[[0.5], [0.5]],
            state_names={'St+1':['low', 'high']})
        cpd_Xt_1 = TabularCPD(
            variable='Xt+1',
            variable_card=2,
            values=[
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                ],
            evidence=['Mt', 'Et', 'St+1'],
            evidence_card=[2,2,2],
            state_names={
                'Xt+1':['no drug', 'give drug'],
                'Mt':['positive', 'negative'],
                'Et':['wealthy', 'poor'],
                'St+1':['low', 'high']
                })
        cpd_Mt_1 = TabularCPD(
            variable='Mt+1',
            variable_card=2,
            values=[
                [0.3, 0.8, 0.7, 0.8],
                [0.7, 0.2, 0.3, 0.2]
                ],
            evidence=['Xt+1', 'Mt'],
            evidence_card=[2,2],
            state_names={
                'Mt+1':['positive', 'negative'],
                'Xt+1':['no drug', 'give drug'],
                'Mt':['positive', 'negative']
                })
        cpd_Et_1 = TabularCPD(
            variable='Et+1',
            variable_card=2,
            values=[
                [0.7, 0.8, 0.3, 0.8],
                [0.3, 0.2, 0.7, 0.2]
                ],
            evidence=['Xt+1', 'Et'],
            evidence_card=[2,2],
            state_names={
                'Et+1':['wealthy', 'poor'],
                'Xt+1':['no drug', 'give drug'],
                'Et':['wealthy', 'poor']
                })
        cpd_Yt_1 = TabularCPD(
            variable='Yt+1',
            variable_card=2,
            values=[
                [0.8, 0.1, 0.1, 0.8, 0.2, 0.7, 0.7, 0.2, 0.3, 0.8, 0.8, 0.3, 0.9, 0.2, 0.2, 0.9],
                [0.2, 0.9, 0.9, 0.2, 0.8, 0.3, 0.3, 0.8, 0.7, 0.2, 0.2, 0.7, 0.1, 0.8, 0.8, 0.1]
                ],
            evidence=['St+1', 'Mt+1', 'Et+1', 'Xt+1'],
            evidence_card=[2, 2, 2, 2],
            state_names={
                'Yt+1':['not healthy', 'healthy'],
                'St+1':['low', 'high'],
                'Mt+1':['positive', 'negative'],
                'Et+1':['wealthy', 'poor'],
                'Xt+1':['no drug', 'give drug']
                })

        self._causal_model.add_cpds(
            cpd_Mt, cpd_Et, cpd_St_1, cpd_Xt_1, cpd_Mt_1, cpd_Et_1, cpd_Yt_1)
        
        self._causal_model.check_model()

    def get_target(self):
        return 'Yt+1'
    
    def get_evidence(self, state):
        return {
            'St+1': self._state[0],
            'Mt': self._state[1],
            'Et': self._state[2],
        }

    def get_action(self):
        return 'Xt+1'
