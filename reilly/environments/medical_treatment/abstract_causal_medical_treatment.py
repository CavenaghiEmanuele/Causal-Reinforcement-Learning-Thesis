from abc import ABC, abstractmethod

import matplotlib as plt
import networkx as nx

from ..causal_environment import CausalEnvironment

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
class AbstractCausalMedicalTreatment(CausalEnvironment, ABC):
   
    @abstractmethod
    def build_causal_model(self):
        pass

    def get_causal_model(self):
        return self._causal_model

    def get_target(self):
        return 'Y'

    def get_good_target_value(self):
        return 'healthy'
    
    @abstractmethod
    def get_evidence(self, state):
        pass

    def get_action(self):
        return 'X'

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

    def get_agent_intent(self):
        return 0
