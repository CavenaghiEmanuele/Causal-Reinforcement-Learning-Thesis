import reilly as rl

import numpy as np
import pandas as pd


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
    list_env = ['confounder_directly_influencing_state_outcome']
    observe_confounder = True
    max_steps = 1000

    alpha = 0.005
    epsilon = 1
    epsilon_decay = 0.97
    gamma = 1
    ####################################

    for env_type in list_env:

        if env_type == 'confounder_directly_influencing_outcome':
            env = rl.ConfounderDirectlyInfluencingOutcome(observe_confounder=observe_confounder)
        elif env_type == 'collider':
            env = rl.Collider(observe_confounder=observe_confounder)
        elif env_type == 'confounder_in_time':
            env = rl.ConfounderInTime(observe_confounder=observe_confounder, build_causal_model=True)
        elif env_type == 'confounder_directly_influencing_state_outcome':
            env = rl.ConfounderDirectlyInfluencingStateOutcome(observe_confounder=observe_confounder)
        elif env_type == 'confounder_not_directly_influencing_outcome':
            env = rl.ConfounderNotDirectlyInfluencingOutcome(observe_confounder=observe_confounder)

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
        # A lot of Causal Q-Learning
        ####################################
        for _ in range(30):
            agent = rl.CausalQLearning(
                states=env.states, actions=env.actions,
                alpha=alpha, epsilon=epsilon, epsilon_decay=epsilon_decay, gamma=gamma)

            session = rl.Session(env=env, agent=agent, max_steps=max_steps)

            results.append(
                session.run(
                    episodes=episodes, test_offset=test_offset, test_samples=test_sample, render=False)
            )

            #print(agent._cache_inference)
        '''
        ####################################
        # A lot of Vanilla Q-Learning
        ####################################
        for _ in range(3):
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

        rl.plot(results)
        #rl.plot_mean(results, n=100)
