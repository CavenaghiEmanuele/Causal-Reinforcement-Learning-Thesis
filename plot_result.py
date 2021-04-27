import reilly as rl

import pandas as pd



if __name__ == '__main__':

    env = 'taxi' # medical_treatment, taxi


    if env == 'medical_treatment':
        # confounder_directly_influencing_outcome, 
        # confounder_not_directly_influencing_outcome, 
        # confounder_directly_influencing_state_outcome
        # collider
        # confounder_in_time
        experiment_name = 'confounder_directly_influencing_outcome'
        observe = 'unobserve_confounder' # unobserve_confounder, observe_confounder
        path = 'results/medical_treatment/' + observe + '/' + experiment_name + '.csv'

        #rl.plot_medical_treatment(path)
        rl.plot_medical_treatment_mean(path)

    elif env == 'taxi':
        path = 'results/taxi/Taxi_confounder.csv'
        #rl.plot_taxi(path)
        rl.plot_taxi_mean(path)
