from abc import ABC, abstractmethod

from .environment import Environment


class CausalEnvironment(Environment, ABC):

    @abstractmethod
    def build_causal_model(self):
        pass

    @abstractmethod
    def plot_causal_model(self):
        pass

    @abstractmethod
    def get_causal_model(self):
        pass

    @abstractmethod
    def get_target(self, hierarchical:bool=False):
        pass

    @abstractmethod
    def get_evidence(self, state, hierarchical:bool=False):
        pass

    @abstractmethod
    def get_actions(self, hierarchical:bool=False):
        pass

    @abstractmethod
    def get_action_values(self, action):
        pass

    def get_agent_intent(self):
        pass
    
    @abstractmethod
    def causal_action_to_env_action(self, causal_action):
        pass