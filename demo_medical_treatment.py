from numpy.core.fromnumeric import trace
from reilly.agents.tabular_agents.temporal_difference import q_learning
import reilly as rl

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count


if __name__ == '__main__':

    ####################################
    # PARAMETERS
    ####################################
    episodes = 100
    test_offset = 10
    test_sample = 100

    # base, collider, collider_in_time, confounder_in_time, confounder_no_influence_cause_reward, confounder_no_influence
    env_type = 'confounder_no_influence' 
    observe_confounder = True
    max_steps = 1000

    alpha = 0.005
    epsilon = 1
    epsilon_decay = 0.97
    gamma = 1
    ####################################

    if env_type == 'base':
        env = rl.Base(observe_confounder=observe_confounder)
    elif env_type == 'collider':
        env = rl.Collider(observe_confounder=observe_confounder)
    elif env_type == 'collider_in_time':
        env = rl.ColliderInTime(observe_confounder=observe_confounder)
    elif env_type == 'confounder_in_time':
        env = rl.ConfounderInTime(observe_confounder=observe_confounder)
    elif env_type == 'confounder_no_influence_cause_reward':
        env = rl.ConfounderNoInfluenceCauseReward(observe_confounder=observe_confounder)
    elif env_type == 'confounder_no_influence':
        env = rl.ConfounderNoInfluence(observe_confounder=observe_confounder)

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
    # A lot of Vanilla Q-Learning
    ####################################
    for _ in range(2):
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

    # Save results to csv
    results.to_csv('results/' + env_type + '.csv', index=False)

    #rl.plot(results)
    #rl.plot_mean(results, n=100)
    rl.plot_results(results)
