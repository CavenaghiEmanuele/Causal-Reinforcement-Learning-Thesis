import numpy as np

from typing import Dict, List

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD

from .medical_treatment import MedicalTreatment

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
class MedicalTreatmentCollider(MedicalTreatment):

    _done: bool
    _state: List[int]
    _step: int
    _max_step: int
    _observe_confounder: bool
    _reward_probs: Dict
    _next_state_probs: Dict
    _next_M_probs: Dict
    _next_E_probs: Dict

    def __init__(
        self, 
        build_causal_model:bool=False, 
        observe_confounder:bool=True, 
        max_steps:int=100, 
        reward_probs:Dict=None
        ):

        self._done = False
        self._observe_confounder = observe_confounder
        self._state = [1, 0, 0]
        self._max_step = max_steps

        if reward_probs == None:
            # P(R=1) - format: S, M, E, X
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
                '[1, 1, 1, 1]': 0.1,
            }
        else:
            self._reward_probs = reward_probs

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
            '[0, 1]': 0.7,
            '[1, 0]': 0.8,
            '[1, 1]': 0.2,
        }

        # P(E=1) - format: X, Y
        self._next_E_probs = {
            '[0, 0]': 0.9,
            '[0, 1]': 0.3,
            '[1, 0]': 0.2,
            '[1, 1]': 0.8,
        }

        if build_causal_model:
            self.build_causal_model()

        self.reset()

    def _run_step(self, action):
        # Reward computation
        reward = np.random.binomial(size=1, n=1, p= self._reward_probs[str(self._state + [action])])[0] # P(R=1)

        # Next state compute (only S, because E and M are confounders)
        self._state[0] = np.random.binomial(size=1, n=1, p= self._next_state_probs[str(self._state[0:1] + [action])])[0] # P(S=1)

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
        pass

    def get_causal_model(self):
        pass

    def get_target(self, hierarchical:bool=False):
        pass
    
    def get_evidence(self, state, hierarchical:bool=False):
        pass

    def get_actions(self, hierarchical:bool=False):
        pass

    def get_action_values(self, action):
        pass

    def plot_causal_model(self):
        pass

    def causal_action_to_env_action(self, causal_action):
        pass

    def get_agent_intent(self):
        pass
