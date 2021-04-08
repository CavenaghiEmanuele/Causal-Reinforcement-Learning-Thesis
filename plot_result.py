import reilly as rl

import pandas as pd



if __name__ == '__main__':

    # base, confounder_no_influence, confounder_no_influence_cause_reward, collider, confounder_in_time
    experiment_name = 'confounder_no_influence'
    observe = 'unobserve_confounder' # unobserve_confounder, observe_confounder
    data = pd.read_csv('results/' + observe + '/' + experiment_name + '.csv')

    #rl.plot(data)
    #rl.plot_mean(data, n=100)
    rl.plot_medical_treatment_results(data, experiment_name)
