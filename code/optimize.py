import numpy as np
import pandas 
import pickle
import optuna
from optuna.samplers import RandomSampler, GridSampler
from tqdm import tqdm

from snr1_snr2_GA import create_problem
from GeneticAlgorithm import GA

def objective(trial):
    """ Non tuneable parameters """
    budget = 5000
    dimension = 50
    repetitions = 5
    
    """ Tuneable parameters """
    # Size of the genome population
    P = trial.suggest_categorical("P", [10, 20, 50, 100])     
    
    # If proportional selections should be used instead of random selection
    R = trial.suggest_categorical("R", [True, False])               
            
    # If proportional selections should be used instead of random selection
    if R: 
        S = None       
    else: 
        S = trial.suggest_categorical("S", [True, False])               
    
    # The propability of doing crossover of two genomes, if 0 don't use crossover
    C = trial.suggest_categorical("C", [0, 0.5, 0.6, 0.8, 0.95])    
    
    # The number of slices for n-crossover, if 0 use uniform crossover
    if C>0: 
        N = trial.suggest_categorical("N", [0, 1, 2, 4])        
    else: 
        N=None               
    
    # The propability of doing mutation on a bit of a genome, if 0 don't use mutation
    M = trial.suggest_categorical("M", [0, 0.01, 0.1, 0.33, 0.5])         

    # Keep track of best fitness
    best_fitness = []
    best_genomes = []

    # Run 20 repetitions for F18
    F18, _logger = create_problem(18, dimension)
    for _ in tqdm(range(repetitions), desc="Loading..."):
    # for _ in range(repetitions): 
        GA18 = GA(F18, budget, dimension)
        GA18.setparameters(P, R, S, C, N, M)
        GA18.main()
        best_fitness.append(GA18.best_fitness)
        best_genomes.append(GA18.best_genome)
        F18.reset()
    _logger.close()
    
    return np.average(best_fitness)
    
def tune():
    print("--- Started Tuning ---")
    
    # Creating optuna study
    study_name = "GA-study"
    storage_name = "{}.csv".format(study_name)
    study = optuna.create_study(sampler=RandomSampler(), direction="maximize", study_name=study_name)

    # Optimize the objective function
    total_trials = 1200
    for _ in range(total_trials):
        study.optimize(objective, n_trials=1)

    # Save results of study
    results = study.trials_dataframe()
    results.to_csv(storage_name)
    
    return results
    
def main():
    # Check if study results already exist, otherwise create new tuning study
    try:
        results = pandas.read_csv('GA-study.csv')
    except:
        results = tune()
        
    # Sort at results
    results = results.sort_values(by=['value'], ascending=False) 
    print(results.columns)
    print(results[['number', 'value', 'params_C', 'params_M', 'params_N', 'params_P', 'params_R', 'params_S', 'state']].head(20))
    
    results1 = results[['value','params_C']].groupby(['params_C']).mean()
    print(results1)
    
    results2 = results[['value','params_M']].groupby(['params_M']).mean()
    print(results2)
    
    results3 = results[['value','params_N']].groupby(['params_N']).mean()
    print(results3)
    
    results4 = results[['value','params_P']].groupby(['params_P']).mean()
    print(results4)

    results5 = results[['value','params_S']].groupby(['params_S']).mean()
    print(results5)
    
if __name__ == "__main__":
    main()