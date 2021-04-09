import gym
import networkx as nx
import pylab as plt
import random
import numpy as np

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD

from .abstract_gym import GymEnvironment
from ..causal_environment import CausalEnvironment


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
class Taxi(GymEnvironment, CausalEnvironment):

    _done: bool
    _subgoal: int
    _passenger_on_taxi: bool
    _confounders: bool
    _thief: bool
    _intent: int # 0 = lead passenger, 1 = call police

    def __init__(self, build_causal_model:bool=False, confounders:bool=False):
        self._env = gym.make('Taxi-v3')
        self._done = False
        self._subgoal = 0
        self._passenger_on_taxi = False

        self._confounders = confounders
        self._thief = False
        self._intent = 0
        if build_causal_model:
            self.build_causal_model()

        self.reset()

    @property
    def actions(self) -> int:
        if self._confounders:
            return self._env.action_space.n + 1
        return self._env.action_space.n

    def run_step(self, action, *args, **kwargs):

        info = {'wins': 0}

        if action == 6: # call police
            next_state, _, _, _ = self._env.step(4) # Pickup only to take the same state, we don't save reward and done
            reward, done = -10, False
            if self._thief:
                reward, done = 20, True
        else: 
            next_state, reward, done, _ = self._env.step(action)
            if self._thief and done and reward == 20:
                reward = -1000
                info['wins'] = 0

        agent_reward = reward
        if done and reward == 20:
            info['wins'] = 1
        self._done == done

        return next_state, reward, agent_reward, done, info

    def reset(self, *args, **kwargs) -> int:
        if self._confounders: 
            self._thief = bool(random.getrandbits(1))
            if self._thief: self._intent = 1
            else: self._intent = 0
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

        if self._confounders:
            
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

    def get_target(self):
        return 'G'
    
    def get_evidence(self, state):

        state = self.decode(state)
        r = {'CP' : 'cab state ' + str(state[0]*5 + state[1])}

        if self._confounders:
            r.update({'thief': str(bool(self._intent))})

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

    def get_actions(self):
        actions = ['P', 'D']
        if self._confounders:
            actions.append('callP')
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
