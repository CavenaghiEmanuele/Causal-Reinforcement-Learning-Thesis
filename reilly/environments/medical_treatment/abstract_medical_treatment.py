from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


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
class AbstractMedicalTreatment(ABC):

    _done: bool
    _state: List[int]
    _step: int
    _max_step: int
    _observe_confounder: bool
    _reward_probs: Dict

    def __init__(
        self, 
        build_causal_model:bool=False, 
        observe_confounder:bool=True, 
        max_steps:int=100, 
        reward_probs:Dict=None
        ):

        self._observe_confounder = observe_confounder
        self._max_step = max_steps
        self._reward_probs = reward_probs

        self.reset()

    @property
    def actions(self) -> int:
        return 2

    @property
    def states(self) -> int:
        if self._observe_confounder:
            return 8
        else:
            return 2
    
    def run_step(self, action, *args, **kwargs):
        return self._run_step(action)
 
    def decode(self, state:int) -> Tuple:
        pass

    def encode(self, state:Tuple) -> int:
        return state[0] + state[1]*2 + state[2]*4
    
    def render(self) -> None:
        pass

    @property
    def probability_distribution(self):
        pass

    @abstractmethod
    def reset(self, *args, **kwargs) -> int:
        pass
   
    @abstractmethod
    def _run_step(self, action):
        pass
