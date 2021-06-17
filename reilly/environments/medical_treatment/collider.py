import random
import numpy as np
from typing import Dict
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel

from .abstract_medical_treatment import AbstractMedicalTreatment
from .abstract_causal_medical_treatment import AbstractCausalMedicalTreatment


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
class Collider(AbstractMedicalTreatment, AbstractCausalMedicalTreatment):

    _next_state_probs: Dict
    _next_M_probs: Dict
    _next_E_probs: Dict

    def __init__(self, build_causal_model:bool=False, observe_confounder:bool=True, max_steps:int=100):

        # P(R=1) - format: S, X
        reward_probs = {
            '[0, 0]': 0.2,
            '[0, 1]': 0.9,
            '[1, 0]': 0.9,
            '[1, 1]': 0.2,
            }

        # P(S=1) - format: X, S
        self._next_state_probs = {
            '[0, 0]': 0.1,
            '[0, 1]': 0.7,
            '[1, 0]': 0.3,
            '[1, 1]': 0.2,
        }

        # P(M=1) - format: X, Y
        self._next_M_probs = {
            '[0, 0]': 0.1,
            '[0, 1]': 0.9,
            '[1, 0]': 0.9,
            '[1, 1]': 0.1,
        }

        # P(E=1) - format: X, Y
        self._next_E_probs = {
            '[0, 0]': 0.7,
            '[0, 1]': 0.3,
            '[1, 0]': 0.3,
            '[1, 1]': 0.7,
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
        # Reward computation
        reward = np.random.binomial(size=1, n=1, p= self._reward_probs[str([self._state[0], action])])[0] # P(R=1)

        # Next state compute
        self._state[0] = np.random.binomial(size=1, n=1, p= self._next_state_probs[str([action, self._state[0]])])[0] # P(S=1)

        # Compute M = self._state[1]
        self._state[1] = np.random.binomial(size=1, n=1, p= self._next_M_probs[str([action, reward])])[0] # P(M=1)

        # Compute E = self._state[2]
        self._state[2] = np.random.binomial(size=1, n=1, p= self._next_E_probs[str([action, reward])])[0] # P(E=1)

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
                ('S', 'X'),
                ('S', 'Y'),

                ('X', 'E'),
                ('X', 'M'),

                ('Y', 'E'),
                ('Y', 'M'),

                ('X', 'Y'),
            ])

        cpd_M = TabularCPD(
            variable='M',
            variable_card=2,
            values=[
                [0.1, 0.9, 0.9, 0.1], 
                [0.9, 0.1, 0.1, 0.9]
                ],
            evidence=['X', 'Y'],
            evidence_card=[2, 2],
            state_names={
                'M':['positive', 'negative'],
                'X':['no drug', 'give drug'],
                'Y':['not healthy', 'healthy'],
                })
        cpd_E = TabularCPD(
            variable='E',
            variable_card=2,
            values=[
                [0.7, 0.3, 0.3, 0.7], 
                [0.3, 0.7, 0.7, 0.3]
                ],
            evidence=['X', 'Y'],
            evidence_card=[2, 2],
            state_names={
                'E':['wealthy', 'poor'],
                'X':['no drug', 'give drug'],
                'Y':['not healthy', 'healthy'],
                })
        
        cpd_S = TabularCPD(
            variable='S',
            variable_card=2,
            values=[[0.5], [0.5]],
            state_names={'S':['low', 'high']})

        cpd_X = TabularCPD(
            variable='X',
            variable_card=2,
            values=[
                [0.5, 0.5],
                [0.5, 0.5]
                ],
            evidence=['S'],
            evidence_card=[2],
            state_names={
                'X':['no drug', 'give drug'],
                'S':['low', 'high']
                })

        cpd_Y = TabularCPD(
            variable='Y',
            variable_card=2,
            values=[
                [0.8, 0.2, 0.2, 0.8],
                [0.2, 0.8, 0.8, 0.2]
                ],
            evidence=['S', 'X'],
            evidence_card=[2, 2],
            state_names={
                'Y':['not healthy', 'healthy'],
                'S':['low', 'high'],
                'X':['no drug', 'give drug']
                })

        self._causal_model.add_cpds(
            cpd_M, cpd_E, cpd_S, cpd_X, cpd_Y)
        
        self._causal_model.check_model()

    def get_evidence(self, state):
        if self._observe_confounder:
            return {
                'S': self._state[0],
                'M': self._state[1],
                'E': self._state[2],
            }
        else:
            return {'S': self._state[0]}