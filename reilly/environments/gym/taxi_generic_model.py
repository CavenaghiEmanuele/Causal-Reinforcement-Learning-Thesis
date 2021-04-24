import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel

import gym

from ..causal_environment import CausalEnvironment
from .abstract_gym import GymEnvironment

'''
Actions:
There are 6 discrete deterministic actions:
- 0: move south
- 1: move north
- 2: move east
- 3: move west
- 4: pickup passenger
- 5: drop off passenger
'''
class TaxiGenericModel(GymEnvironment, CausalEnvironment):

    _observe_confounders: bool

    def __init__(self, build_causal_model:bool=False, observe_confounders:bool=False):
        self._env = gym.make('Taxi-v3')
        self._observe_confounders = observe_confounders
        if build_causal_model:
            self.build_causal_model()
        self.reset()

    def run_step(self, action, *args, **kwargs):
        info = {'wins': 0}
        next_state, reward, done, _ = self._env.step(action)
        if done and reward == 20:
            info['wins'] = 1
        return next_state, reward, done, info

    def reset(self, *args, **kwargs) -> int:
        return self._env.reset()

    def decode(self, state):
        return tuple(self._env.decode(state))

    ###############################################
    # Causal section
    ###############################################

    def build_causal_model(self):
        #################################
        # Defining the model structure
        #################################
        # PP = Passenger Position
        # CP = Cab Position
        # DP = Destination Position of the passenger
        # onPP = the cab is on the Passenger Position
        # onDP = the cab is on the Destination Position
        # inC = passenger is in the cab

        self._causal_model = BayesianModel(
            [
                ('PP', 'onPP'),
                ('CP', 'onPP'),
                ('CP', 'onDP'),
                ('DP', 'onDP'),
            
                ('inC', 'X'),
                ('onPP', 'X'),
                ('onDP', 'X'),

                ('onPP', 'Y'),
                ('onDP', 'Y'),
                ('inC', 'Y'),
                ('X', 'Y'),
            ]
        )

        # Defining individual CPDs.
        cpd_PP = TabularCPD(
            variable='PP', 
            variable_card=25, 
            values=[[0.04] for _ in range(0,25)], #All states have the same probability
            state_names={'PP': ['state ' + str(i) for i in range(0,25)]}
            )
        cpd_CP = TabularCPD(
            variable='CP', 
            variable_card=25, 
            values=[[0.04] for _ in range(0,25)], #All states have the same probability
            state_names={'CP': ['cab state ' + str(i) for i in range(0,25)]}
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
        cpd_inC = TabularCPD(
            variable='inC', 
            variable_card=2, 
            values=[[0.5], [0.5]],
            state_names={'inC': ['False', 'True']}
            )       
        cpd_X = TabularCPD(
            variable='X', 
            variable_card=2, 
            values=[
                [0.5, 0.5, 0.5, 0, 1, 0.5, 1, 0], 
                [0.5, 0.5, 0.5, 1, 0, 0.5, 0, 1]
            ],
            evidence=['onPP', 'onDP', 'inC'],
            evidence_card=[2, 2, 2],
            state_names={
                'X': ['Pickup', 'Dropoff'],
                'onPP': ['False', 'True'],
                'onDP': ['False', 'True'],
                'inC': ['False', 'True']
                }
            )
        cpd_Y = TabularCPD(
            variable='Y',
            variable_card=2,
            values=[
                [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0], 
                [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1]
            ],
            evidence=['X', 'inC', 'onDP', 'onPP'],
            evidence_card=[2, 2, 2, 2],
            state_names={
                'Y': ['False', 'True'],
                'X': ['Pickup', 'Dropoff'],
                'inC': ['False', 'True'],
                'onDP': ['False', 'True'],
                'onPP': ['False', 'True']
                }
                )

        # Associating the CPDs with the network
        self._causal_model.add_cpds(cpd_PP, cpd_DP, cpd_CP, cpd_onPP, cpd_onDP, cpd_inC, cpd_X, cpd_Y)

        # check_model checks for the network structure and CPDs and verifies that the CPDs are correctly 
        # defined and sum to 1.
        self._causal_model.check_model()

    def get_causal_model(self):
        return self._causal_model

    def get_target(self):
        return 'Y'

    def get_evidence(self, state):
        # PP = Passenger Position
        # CP = Cab Position
        # DP = Destination Position of the passenger
        # onPP = the cab is on the Passenger Position
        # onDP = the cab is on the Destination Position
        # inC = passenger is in the cab

        state = self.decode(state)
        evidence = {'CP' : 'cab state ' + str(state[0]*5 + state[1])}

        pp = {
            0 : {'PP' : 'state ' + str(0), 'inC' : 'False'},
            1 : {'PP' : 'state ' + str(4), 'inC' : 'False'},
            2 : {'PP' : 'state ' + str(20), 'inC' : 'False'},
            3 : {'PP' : 'state ' + str(23), 'inC' : 'False'},
            4 : {'PP' : 'state ' + str(state[0]*5 + state[1]), 'inC' : 'True'}
        }
        evidence.update(pp[state[2]])

        pd = {
            0 : {'DP' : 'destination ' + str(0)},
            1 : {'DP' : 'destination ' + str(4)},
            2 : {'DP' : 'destination ' + str(20)},
            3 : {'DP' : 'destination ' + str(23)}
        }  
        evidence.update(pd[state[3]])
        return evidence

    def get_action(self):
        return 'X'

    def get_action_values(self):
        return ['Pickup', 'Dropoff']

    def get_good_target_value(self):
        return 'True'

    def causal_action_to_env_action(self, causal_action):
        if causal_action == 'Pickup':
            return 4
        elif causal_action == 'Dropoff':
            return 5
