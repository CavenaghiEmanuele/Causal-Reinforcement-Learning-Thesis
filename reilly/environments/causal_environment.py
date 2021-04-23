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
    def get_target(self):
        pass

    @abstractmethod
    def get_evidence(self, state):
        pass

    @abstractmethod
    def get_action(self):
        pass

    @abstractmethod
    def get_action_values(self):
        pass
    
    def get_good_target_value(self):
        pass

    @abstractmethod
    def causal_action_to_env_action(self, causal_action):
        pass
