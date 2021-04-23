import reilly as rl

import pandas as pd



if __name__ == '__main__':

    # confounder_directly_influencing_outcome, 
    # confounder_not_directly_influencing_outcome, 
    # confounder_directly_influencing_state_outcome
    # collider
    # confounder_in_time

    experiment_name = 'confounder_directly_influencing_outcome'
    observe = 'observe_confounder' # unobserve_confounder, observe_confounder
    #data = pd.read_csv('results/' + observe + '/' + experiment_name + '.csv')

    data = pd.read_csv('results/' + experiment_name + '.csv')

    #rl.plot_mean(data, n=100)
    rl.plot_medical_treatment_results(data, experiment_name)
