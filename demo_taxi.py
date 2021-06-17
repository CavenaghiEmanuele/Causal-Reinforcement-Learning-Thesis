import pandas as pd

import reilly as rl


if __name__ == '__main__':

    ####################################
    # PARAMETERS
    ####################################
    episodes = 2000
    test_offset = 100
    test_sample = 30

    env_name = 'taxi_confounder' # taxi, taxi_generic_model, taxi_confounder
    observe_confounders = True
    number_of_agents = 1

    alpha = 0.3
    epsilon = 1
    epsilon_decay = 0.995
    gamma = 0.98
    min_epsilon = 0.05
    causal_threshold = 0.6
    ####################################

    if env_name == 'taxi':
        env = rl.Taxi(build_causal_model=True, confounders=observe_confounders)
    elif env_name == 'taxi_generic_model':
        env = rl.TaxiGenericModel(build_causal_model=True, observability='partial') # full, partial
    elif env_name == 'taxi_confounder':
        env = rl.TaxiConfounder(build_causal_model=True, observe_confounder=observe_confounders)

    results = []

    ####################################
    # Random agent
    ####################################
    random_agents = [rl.Random(actions=env.actions) for _ in range(number_of_agents)]
    for agent in random_agents:
        session = rl.Session(env=env, agent=agent)
        results.append(
            session.run(
                episodes=episodes, test_offset=test_offset, test_samples=test_sample, render=False)
        )           
    
    ####################################
    # Causal Q-Learning
    ####################################
    causal_q_learning_agents = [
        rl.CausalQLearning(states=env.states, actions=env.actions,alpha=alpha, 
            epsilon=epsilon, epsilon_decay=epsilon_decay, gamma=gamma, min_epsilon=min_epsilon, causal_threshold=causal_threshold) 
        for _ in range(number_of_agents)]
        
    for agent in causal_q_learning_agents:
        session = rl.Session(env=env, agent=agent)
        results.append(
            session.run(
                episodes=episodes, test_offset=test_offset, test_samples=test_sample, render=False)
        )

    ####################################
    # Vanilla Q-Learning
    ####################################
    q_learning_agents = [
        rl.QLearning(
            states=env.states, actions=env.actions,
            alpha=alpha, epsilon=epsilon, epsilon_decay=epsilon_decay, gamma=gamma, min_epsilon=min_epsilon)
            for _ in range(number_of_agents)]
        
    for agent in q_learning_agents:
        session = rl.Session(env=env, agent=agent)
        results.append(
            session.run(
                episodes=episodes, test_offset=test_offset, test_samples=test_sample, render=False)
        )

    ####################################
    # Save results
    ####################################
    results = pd.concat(results)
    results.to_csv('results/' + env_name + '.csv', index=False)
