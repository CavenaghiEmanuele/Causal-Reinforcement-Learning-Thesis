from reilly.agents.agent import Agent
import pandas as pd

import reilly as rl


if __name__ == '__main__':

    ####################################
    # PARAMETERS
    ####################################
    episodes = 2000
    test_offset = 100
    test_sample = 50

    env_name = 'taxi_confounder' # taxi, taxi_generic_model, taxi_confounder
    observe_confounders = True

    alpha = 0.1
    epsilon = 1
    epsilon_decay = 0.99
    gamma = 0.99
    min_epsilon = 0.01
    causal_threshold = 0.5
    ####################################

    if env_name == 'taxi':
        env = rl.Taxi(build_causal_model=True, confounders=observe_confounders)
    elif env_name == 'taxi_generic_model':
        env = rl.TaxiGenericModel(build_causal_model=True)
    elif env_name == 'taxi_confounder':
        env = rl.TaxiConfounder(build_causal_model=True, observe_confounder=observe_confounders)

    results = []
    '''
    ####################################
    # Vanilla Q-Learning
    ####################################
    agent = rl.QLearning(
        states=env.states, actions=env.actions,
        alpha=alpha, epsilon=epsilon, epsilon_decay=epsilon_decay, gamma=gamma, min_epsilon=min_epsilon)

    session = rl.Session(env=env, agent=agent)

    results.append(
        session.run(
            episodes=episodes, test_offset=test_offset, test_samples=test_sample, render=False)
    )
    ''''''
    ####################################
    # Random agent
    ####################################
    random_agent = rl.Random(actions=env.actions)
    session = rl.Session(env=env, agent=random_agent)
    results.append(
        session.run(
            episodes=episodes, test_offset=test_offset, test_samples=test_sample, render=False)
    )
    '''
    ####################################
    # Causal Q-Learning
    ####################################
    causal_agent = rl.CausalQLearning(
        states=env.states, actions=env.actions,
        alpha=alpha, epsilon=epsilon, epsilon_decay=epsilon_decay, gamma=gamma, 
        min_epsilon=min_epsilon, causal_threshold=causal_threshold)

    session = rl.Session(env=env, agent=causal_agent)
    
    results.append(
        session.run(
            episodes=episodes, test_offset=test_offset, test_samples=test_sample, render=False)
    )

    ####################################
    # PLOT
    ####################################
   
    results = pd.concat(results)

    results.to_csv('tmp.csv', index=False)
    rl.plot(results)   
