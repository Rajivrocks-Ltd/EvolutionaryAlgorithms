import optuna
from optuna.samplers import RandomSampler, GridSampler
from snr1_snr2_GA import create_problem
from GeneticAlgorithm import GA
import numpy as np
import pandas 
import pickle

def objective(trial):
    # Non tuneable parameters
    budget = 5000
    dimension = 50
    repetitions = 10
    
    # Tuneable parameters
    P = trial.suggest_categorical("P", [6, 10, 20, 50])             # Size of the genome population
    S = trial.suggest_categorical("S", [True, False])               # If proportional selections should be used instead of random selection
    C = trial.suggest_categorical("C", [0, 0.5, 0.6, 0.8, 0.95])    # The propability of doing crossover of two genomes, if 0 don't use crossover
    N = trial.suggest_categorical("N", [0, 1, 2, 4])                # The number of slices for n-crossover, if 0 use uniform crossover
    M = trial.suggest_categorical("M", [0, 0.1, 0.33, 0.5])         # The propability of doing mutation on a bit of a genome, if 0 don't use mutation

    # Keep track of best fitness
    best_fitnesses = []

    # this how you run your algorithm with 20 repetitions/independent run
    F18, _logger = create_problem(18, dimension)
    for rep in range(repetitions): 
        GA18 = GA(F18, budget, dimension)
        GA18.setparameters(P, S, C, N, M)
        GA18.main()
        best_fitnesses.append(GA18.best_fitness)
        F18.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder

    return np.average(best_fitnesses)

study_name = "GA-study"
storage_name = "/code/{}.pkl".format(study_name)
study = optuna.create_study(sampler=RandomSampler(), direction="maximize", study_name=study_name)

# Optimize the objective function.
total_trials = 640
for _ in range(total_trials):
    # try:
    #     # Load existing study
    #     study = pickle.load(open('{}'.format(storage_name), 'rb'))
    # except:
    #     pass

    # try:
    # Optimize new trial
    study.optimize(objective, n_trials=1)
    # except:
    #     pass

    # Save study
    # pickle.dump(study)

results = study.trials_dataframe().sort_values(by=['value'], ascending=False)
# print(results.columns)
print(results.loc[:,['number', 'value', 'params_C', 'params_M', 'params_N', 'params_P', 'params_S', 'state']])