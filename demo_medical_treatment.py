from networkx.generators.triads import TRIAD_EDGES
from reilly.environments import causal_environment
import reilly as rl

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pprint


if __name__ == '__main__':

    ####################################
    # PARAMETERS
    ####################################
    episodes = 200
    test_offset = 10
    test_sample = 200

    env_type = 'collider' # base, collider
    observe_confounder = True
    max_steps = 1000
    
    alpha = 0.05
    epsilon = 0.075
    epsilon_decay = 0.99
    gamma = 0.99
    ####################################

    if env_type == 'base':
        env = rl.MedicalTreatment(observe_confounder=observe_confounder)
    elif env_type == 'collider':
        env = rl.MedicalTreatmentCollider(observe_confounder=observe_confounder)

    results = []

    ####################################
    # Random Agent
    ####################################
    random_agent = rl.Random(actions=env.actions)

    session = rl.Session(env=env, agent=random_agent, max_steps=max_steps)

    results.append(
        session.run(
            episodes=episodes, test_offset=test_offset, test_samples=test_sample, render=False)
    )

    ####################################
    # Vanilla Q-Learning
    ####################################
    agent = rl.QLearning(
        states=env.states, actions=env.actions,
        alpha=alpha, epsilon=epsilon, epsilon_decay=epsilon_decay, gamma=gamma)

    session = rl.Session(env=env, agent=agent, max_steps=max_steps)

    results.append(
        session.run(
            episodes=episodes, test_offset=test_offset, test_samples=test_sample, render=False)
    )
    
    ####################################
    # PLOT
    ####################################
    results = pd.concat(results)

    rl.plot(results)   
