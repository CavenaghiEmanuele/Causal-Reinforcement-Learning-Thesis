from abc import ABC, abstractmethod
from .environment import Environment


class HierarchicalEnvironment(Environment, ABC):

    @property
    @abstractmethod
    def states_hierarchical(self) -> int:
        pass

    @property
    @abstractmethod
    def super_states(self) -> int:
        pass

    @property
    @abstractmethod
    def super_actions(self) -> int:
        pass

    @abstractmethod
    def _super_reset(self, *args, **kwargs) -> int:
        pass

    @abstractmethod
    def super_run_step(self, action, *args, **kwargs):
        pass
 
    @abstractmethod
    def super_reach_current_subgoal(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_current_subgoal(self):
        pass
