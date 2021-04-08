import reilly as rl

import pandas as pd



if __name__ == '__main__':

    # base, collider, collider_in_time, confounder_in_time, confounder_no_influence_cause_reward, confounder_no_influence
    experiment_name = 'base'
    data = pd.read_csv('results/' + experiment_name + '.csv')

    #rl.plot(data)
    #rl.plot_mean(data, n=100)
    rl.plot_medical_treatment_results(data, experiment_name)
