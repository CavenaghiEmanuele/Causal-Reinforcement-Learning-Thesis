import networkx as nx
from numpy.lib.utils import info
import pylab as plt
import random
import numpy as np

from typing import List, Tuple

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD

from .causal_environment import CausalEnvironment


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

class MedicalTreatment(CausalEnvironment):

    _done: bool
    _state: List[int]
    _step: int
    _max_step: int
    _observe_confounder: bool

    def __init__(self, build_causal_model:bool=False, observe_confounder:bool=True, max_steps:int=100):
        self._done = False
        self._observe_confounder = observe_confounder
        self._state = [1, 0, 0]
        self._max_step = max_steps

        if build_causal_model:
            self.build_causal_model()

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
        info = {}
        next_state, reward, done, _ = self._run_step(action)
        agent_reward = reward

        return next_state, reward, agent_reward, done, info

    def reset(self, *args, **kwargs) -> int:
        self._state = [1, random.randint(0,1), random.randint(0,1)]
        self._step = 0
        self._done = False
        if self._observe_confounder:
            return self.encode(self._state)
        else:
            return self._state[0]
    
    def decode(self, state:int) -> Tuple:
        pass

    def encode(self, state:Tuple) -> int:
        return state[0] + state[1]*2 + state[2]*4

    def _run_step(self, action):
        S = self._state[0]
        M = self._state[1]
        E = self._state[2]

        # Next state compute
        if action == 0 and S == 0:
            self._state[0] = np.random.binomial(size=1, n=1, p= 0.1)[0] # P(S=1)
        elif action == 0 and S == 1:
            self._state[0] = np.random.binomial(size=1, n=1, p= 0.7)[0] # P(S=1)
        elif action == 1 and S == 0:
            self._state[0] = np.random.binomial(size=1, n=1, p= 0.3)[0] # P(S=1)
        elif action == 1 and S == 1:
            self._state[0] = np.random.binomial(size=1, n=1, p= 0.2)[0] # P(S=1)

        # Reward computation
        reward = 0
        if S == 0:
            if M == 0:
                if E == 0:
                    if action == 0:
                        reward = np.random.binomial(size=1, n=1, p= 0.2)[0] # P(R=1)                    
                    else: # action == 1
                        reward = np.random.binomial(size=1, n=1, p= 0.9)[0] # P(R=1)
                else: # E == 1
                    if action == 0:
                        reward = np.random.binomial(size=1, n=1, p= 0.9)[0] # P(R=1)
                    else: # action == 1
                        reward = np.random.binomial(size=1, n=1, p= 0.2)[0] # P(R=1)
            else: # M == 1
                if E == 0:
                    if action == 0:
                        reward = np.random.binomial(size=1, n=1, p= 0.8)[0] # P(R=1)
                    else: # action == 1
                        reward = np.random.binomial(size=1, n=1, p= 0.3)[0] # P(R=1)
                else: # E == 1
                    if action == 0:
                        reward = np.random.binomial(size=1, n=1, p= 0.3)[0] # P(R=1)
                    else: # action == 1
                        reward = np.random.binomial(size=1, n=1, p= 0.8)[0] # P(R=1)
        else: #S == 1
            if M == 0:
                if E == 0:
                    if action == 0:
                        reward = np.random.binomial(size=1, n=1, p= 0.7)[0] # P(R=1)
                    else: # action == 1
                        reward = np.random.binomial(size=1, n=1, p= 0.2)[0] # P(R=1)
                else: # E == 1
                    if action == 0:
                        reward = np.random.binomial(size=1, n=1, p= 0.2)[0] # P(R=1)
                    else: # action == 1
                        reward = np.random.binomial(size=1, n=1, p= 0.7)[0] # P(R=1)
            else: # M == 1
                if E == 0:
                    if action == 0:
                        reward = np.random.binomial(size=1, n=1, p= 0.1)[0] # P(R=1)
                    else: # action == 1
                        reward = np.random.binomial(size=1, n=1, p= 0.8)[0] # P(R=1)
                else: # E == 1
                    if action == 0:
                        reward = np.random.binomial(size=1, n=1, p= 0.8)[0] # P(R=1)                    
                    else: # action == 1
                        reward = np.random.binomial(size=1, n=1, p= 0.1)[0] # P(R=1)

        # Done computation
        done = False
        self._step += 1
        if self._step > self._max_step:
            done = True
        
        if self._observe_confounder:
            return self.encode(self._state), reward, done, {} # info = {}
        else:
            return self._state[0], reward, done, {} # info = {}
    
    def render(self) -> None:
        pass

    @property
    def probability_distribution(self):
        pass

    ###############################################
    # Causal section
    ###############################################
    
    def build_causal_model(self):
        #################################
        # Defining the model structure
        #################################
        # PP = Passenger Position
        # DP = Destination Position of the passenger
        # CP = Cab Position
        # onPP = the cab is on the Passenger Position
        # onDP = the cab is on the Destination Position

        # P = Pickup the passenger
        # D = Dropoff the passenger

        # inC = passenger is in the Cab
        # G = Goal

        # thief = passenger is a thief (confounder)
        # callP = call police (action)

        self._causal_model = BayesianModel(
            [
                ('PP', 'onPP'),
                ('CP', 'onPP'),
                ('CP', 'onDP'),
                ('DP', 'onDP'),
                ('P', 'inC'),
                ('onPP', 'inC'),
                ('D', 'G'),
                ('inC', 'G'),
                ('onDP', 'G')
            ]
        )

        # Defining individual CPDs.
        cpd_PP = TabularCPD(
            variable='PP', 
            variable_card=25, 
            values=[[0.04] for _ in range(0,25)], #All states have the same probability
            state_names={'PP': ['state ' + str(i) for i in range(0,25)]}
            )
        cpd_DP = TabularCPD(
            variable='DP',
            variable_card=25,
            values=[
                [0.25], [0], [0], [0], [0.25],
                [0], [0], [0], [0], [0],
                [0], [0], [0], [0], [0],
                [0], [0], [0], [0], [0],
                [0.25], [0], [0], [0.25], [0],        
            ],
            state_names={'DP': ['destination ' + str(i) for i in range(0,25)]}
            )
        cpd_CP = TabularCPD(
            variable='CP', 
            variable_card=25, 
            values=[[0.04] for _ in range(0,25)], #All states have the same probability
            state_names={'CP': ['cab state ' + str(i) for i in range(0,25)]}
            )
        cpd_onPP = TabularCPD(
            variable='onPP',
            variable_card=2,
            values=[
                np.ndarray.flatten(np.ones(25) - np.diag(np.ones(25))),
                np.ndarray.flatten(np.diag(np.ones(25)))
            ],
            evidence=['PP', 'CP'],
            evidence_card=[25, 25],
            state_names={
                'onPP': ['False', 'True'], 
                'PP': ['state ' + str(i) for i in range(0,25)],
                'CP': ['cab state ' + str(i) for i in range(0,25)]
                }
            ) 
        cpd_onDP = TabularCPD(
            variable='onDP', 
            variable_card=2, 
            values=[
                np.ndarray.flatten(np.ones(25) - np.diag(np.ones(25))),
                np.ndarray.flatten(np.diag(np.ones(25)))
            ],
            evidence=['DP', 'CP'],
            evidence_card=[25, 25],
            state_names={
                'onDP': ['False', 'True'], 
                'DP': ['destination ' + str(i) for i in range(0,25)],
                'CP': ['cab state ' + str(i) for i in range(0,25)]
                }
            )
        cpd_P = TabularCPD(
            variable='P', 
            variable_card=2, 
            values=[[0.5], [0.5]],
            state_names={'P': ['False', 'True']}
            )
        cpd_inC = TabularCPD(
            variable='inC',
            variable_card=2,
            values=[
                [1, 1, 1, 0], 
                [0, 0, 0, 1]
            ],
            evidence=['P', 'onPP'],
            evidence_card=[2, 2],
            state_names={
                'inC': ['False', 'True'],
                'P': ['False', 'True'],
                'onPP': ['False', 'True']
                }
            )
        cpd_D = TabularCPD(
                variable='D', 
                variable_card=2, 
                values=[[0.5], [0.5]],
                state_names={'D': ['False', 'True']}
            )

        if self._observe_confounders:
            
            self._causal_model.add_edge('thief', 'G')
            self._causal_model.add_edge('callP', 'G')

            cpd_thief = TabularCPD(
                variable='thief',
                variable_card=2,
                values=[[0.5], [0.5]],
                state_names={'thief': ['False', 'True']}
                )       
            cpd_callP = TabularCPD(
                variable='callP',
                variable_card=2,
                values=[[0.5], [0.5]],
                state_names={'callP': ['False', 'True']}
                )
            cpd_G = TabularCPD(
                variable='G',
                variable_card=2,
                values=[
                    [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1 ]
                ],
                evidence=['D', 'inC', 'onDP', 'callP', 'thief'],
                evidence_card=[2, 2, 2, 2, 2],
                state_names={
                    'G': ['False', 'True'],
                    'D': ['False', 'True'],
                    'inC': ['False', 'True'],
                    'onDP': ['False', 'True'],
                    'callP': ['False', 'True'],
                    'thief': ['False', 'True']
                    }
                    )

            self._causal_model.add_cpds(cpd_thief, cpd_callP)

        else: # No confounding
            cpd_G = TabularCPD(
                variable='G',
                variable_card=2,
                values=[
                    [1, 1, 1, 1, 1, 1, 1, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 1]
                ],
                evidence=['D', 'inC', 'onDP'],
                evidence_card=[2, 2, 2],
                state_names={
                    'G': ['False', 'True'],
                    'D': ['False', 'True'],
                    'inC': ['False', 'True'],
                    'onDP': ['False', 'True'],
                    }
                    )

        # Associating the CPDs with the network
        self._causal_model.add_cpds(cpd_PP, cpd_DP, cpd_CP, cpd_onPP, cpd_onDP, cpd_P, cpd_D, cpd_inC, cpd_G)

        # check_model checks for the network structure and CPDs and verifies that the CPDs are correctly 
        # defined and sum to 1.
        self._causal_model.check_model()

    def get_causal_model(self):
        return self._causal_model

    def get_target(self, hierarchical:bool=False):
        if hierarchical and self._subgoal == 1:
            return 'inC'
        return 'G'
    
    def get_evidence(self, state, hierarchical:bool=False):
        if hierarchical:
            while state >= 500:
                state -= self.states

        state = self.decode(state)
        r = {'CP' : 'cab state ' + str(state[0]*5 + state[1])}

        if self._observe_confounders:
            r.update({'thief': str(bool(self._intent))})

        if hierarchical and self._subgoal == 1:
            pp = {
                0 : {'PP' : 'state ' + str(0)},
                1 : {'PP' : 'state ' + str(4)},
                2 : {'PP' : 'state ' + str(20)},
                3 : {'PP' : 'state ' + str(23)},
                4 : {'PP' : 'state ' + str(state[0]*5 + state[1])}
            }
        else:
            pp = {
                0 : {'PP' : 'state ' + str(0), 'inC' : 'False'},
                1 : {'PP' : 'state ' + str(4), 'inC' : 'False'},
                2 : {'PP' : 'state ' + str(20), 'inC' : 'False'},
                3 : {'PP' : 'state ' + str(23), 'inC' : 'False'},
                4 : {'PP' : 'state ' + str(state[0]*5 + state[1]), 'inC' : 'True'}
            }

        r.update(pp[state[2]])
        pd = {
            0 : {'DP' : 'destination ' + str(0)},
            1 : {'DP' : 'destination ' + str(4)},
            2 : {'DP' : 'destination ' + str(20)},
            3 : {'DP' : 'destination ' + str(23)}
        }  
        r.update(pd[state[3]])
        return r

    def get_actions(self, hierarchical:bool=False):
        actions = []
        if self._observe_confounders:
            actions.append('callP')
    
        if hierarchical:
            if self._subgoal == 0:
                actions.extend(['P', 'D'])
            elif self._subgoal == 1:
                actions.append('P')
            elif self._subgoal == 2:
                actions.append('D')
            return actions
        
        actions.extend(['P', 'D'])
        return actions

    def get_action_values(self, action):
        # P: ['False', 'True']
        # D: ['False', 'True']
        #callP: ['False', 'True']
        return ['False', 'True']

    def plot_causal_model(self):
        nx.draw(self._causal_model, with_labels=True)
        plt.show()

    def causal_action_to_env_action(self, causal_action):
        if causal_action == 'P':
            return 4
        elif causal_action == 'D':
            return 5
        elif causal_action == 'callP':
            return 6

    def get_agent_intent(self):
        return self._intent
