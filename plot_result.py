import reilly as rl


if __name__ == '__main__':

    env = 'taxi' # medical_treatment, taxi
    observe = 'observe_confounder' # unobserve_confounder, observe_confounder

    if env == 'medical_treatment':
        # confounder_directly_influencing_outcome, 
        # confounder_not_directly_influencing_outcome, 
        # confounder_directly_influencing_state_outcome
        # collider
        # confounder_in_time
        experiment_name = 'confounder_in_time'
        path = 'results/medical_treatment/' + observe + '/' + experiment_name + '.csv'

        #rl.plot_medical_treatment(path)
        rl.plot_medical_treatment_mean(path)

    elif env == 'taxi':
        experiment_name = 'taxi_confounder' # taxi_generic_model, taxi_confounder
        path = 'results/taxi/' + observe + '/' + experiment_name + '.csv'

        #rl.plot_taxi(path)
        rl.plot_taxi_mean(path)
