import pandas as pd

import reilly as rl

if __name__ == '__main__':

    ####################################
    # PARAMETERS
    ####################################
    episodes = 100
    test_offset = 10
    test_sample = 100

    '''
    list_env = [
        'confounder_directly_influencing_outcome',
        'collider',
        'confounder_in_time',
        'confounder_directly_influencing_state_outcome',
        'confounder_not_directly_influencing_outcome']
    '''
    list_env = ['confounder_directly_influencing_outcome']
    observe_confounder = True
    number_of_agents = 30
    max_steps = 1000

    alpha = 0.005
    epsilon = 1
    epsilon_decay = 0.97
    gamma = 1
    ####################################

    for env_type in list_env:

        if env_type == 'confounder_directly_influencing_outcome':
            env = rl.ConfounderDirectlyInfluencingOutcome(observe_confounder=observe_confounder, build_causal_model=True)
        elif env_type == 'collider':
            env = rl.Collider(observe_confounder=observe_confounder)
        elif env_type == 'confounder_in_time':
            env = rl.ConfounderInTime(observe_confounder=observe_confounder, build_causal_model=True)
        elif env_type == 'confounder_directly_influencing_state_outcome':
            env = rl.ConfounderDirectlyInfluencingStateOutcome(observe_confounder=observe_confounder, build_causal_model=True)
        elif env_type == 'confounder_not_directly_influencing_outcome':
            env = rl.ConfounderNotDirectlyInfluencingOutcome(observe_confounder=observe_confounder)

        results = []
        
        ####################################
        # Random Agent
        ####################################
        random_agents = [rl.Random(actions=env.actions) for _ in range(number_of_agents)]

        for agent in random_agents:
            session = rl.Session(env=env, agent=agent, max_steps=max_steps)
            results.append(
                session.run(
                    episodes=episodes, test_offset=test_offset, test_samples=test_sample, render=False)
            )
        
        ####################################
        # A lot of Causal Q-Learning
        ####################################
        causal_q_learning_agents = [
            rl.CausalQLearning(states=env.states, actions=env.actions,alpha=alpha, epsilon=epsilon, epsilon_decay=epsilon_decay, gamma=gamma) 
            for _ in range(number_of_agents)]
        
        for agent in causal_q_learning_agents:
            session = rl.Session(env=env, agent=agent, max_steps=max_steps)
            results.append(
                session.run(
                    episodes=episodes, test_offset=test_offset, test_samples=test_sample, render=False)
            )
        
        ####################################
        # A lot of Vanilla Q-Learning
        ####################################
        q_learning_agents = [
            rl.QLearning(states=env.states, actions=env.actions,alpha=alpha, epsilon=epsilon, epsilon_decay=epsilon_decay, gamma=gamma) 
            for _ in range(number_of_agents)]
        
        for agent in q_learning_agents:
            session = rl.Session(env=env, agent=agent, max_steps=max_steps)
            results.append(
                session.run(
                    episodes=episodes, test_offset=test_offset, test_samples=test_sample, render=False)
            )

        ####################################
        # Save results
        ####################################
        results = pd.concat(results)
        results.to_csv('results/' + env_type + '.csv', index=False)
