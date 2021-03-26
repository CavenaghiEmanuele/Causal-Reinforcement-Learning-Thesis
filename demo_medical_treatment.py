from reilly.agents.tabular_agents.temporal_difference import q_learning
import reilly as rl

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count


def _run_q_learning(fake):
    agent = rl.QLearning(
        states=env.states, actions=env.actions,
        alpha=alpha, epsilon=epsilon, epsilon_decay=epsilon_decay, gamma=gamma)

    session = rl.Session(env=env, agent=agent, max_steps=max_steps)
    return session.run(episodes=episodes, test_offset=test_offset, test_samples=test_sample, render=False)


if __name__ == '__main__':
    '''
    reward_probs = {
                '[0, 0, 0, 0]': 0.2,
                '[0, 0, 0, 1]': 0.9,
                '[0, 0, 1, 0]': 0.9,
                '[0, 0, 1, 1]': 0.2,
                '[0, 1, 0, 0]': 0.9,
                '[0, 1, 0, 1]': 0.2,
                '[0, 1, 1, 0]': 0.2,
                '[0, 1, 1, 1]': 0.9,

                '[1, 0, 0, 0]': 0.8,
                '[1, 0, 0, 1]': 0.1,
                '[1, 0, 1, 0]': 0.1,
                '[1, 0, 1, 1]': 0.8,
                '[1, 1, 0, 0]': 0.1,
                '[1, 1, 0, 1]': 0.8,
                '[1, 1, 1, 0]': 0.8,
                '[1, 1, 1, 1]': 0.1,
            }
    env = rl.MedicalTreatment(observe_confounder=True, reward_probs=reward_probs)
    '''
    ####################################
    # PARAMETERS
    ####################################
    episodes = 30
    test_offset = 2
    test_sample = 100

    env_type = 'bigger_collider' # base, collider, bigger_collider
    observe_confounder = False
    max_steps = 1000

    alpha = 0.1
    epsilon = 0.1
    epsilon_decay = 0.99
    gamma = 1
    ####################################

    if env_type == 'base':
        env = rl.MedicalTreatment(observe_confounder=observe_confounder)
    elif env_type == 'collider':
        env = rl.MedicalTreatmentCollider(observe_confounder=observe_confounder)
    elif env_type == 'bigger_collider':
        env = rl.MedicalTreatmentBiggerCollider(observe_confounder=observe_confounder)

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
    '''
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
    '''
    
    ####################################
    # A lot of Vanilla Q-Learning
    ####################################
    for _ in range(9):
        agent = rl.QLearning(
            states=env.states, actions=env.actions,
            alpha=alpha, epsilon=epsilon, epsilon_decay=epsilon_decay, gamma=gamma)

        session = rl.Session(env=env, agent=agent, max_steps=max_steps)

        results.append(
            session.run(
                episodes=episodes, test_offset=test_offset, test_samples=test_sample, render=False)
        )
    
    '''
    parms = [i for i in range(100)]
    pool = Pool(cpu_count())
    multithread_result = pool.map(_run_q_learning, parms)
    pool.close()
    pool.join()

    results.extend(multithread_result)
    '''
    ####################################
    # PLOT
    ####################################
    results = pd.concat(results)

    rl.plot(results)
    #rl.plot_mean(results, n=100)
