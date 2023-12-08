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
    problem = 19
    
    """ Tuneable parameters """
    if problem == 18:
        # Size of the genome population
        P = trial.suggest_categorical("P", [10, 20, 36, 50, 76, 100])  
        
        # Type of selection
        S = trial.suggest_categorical("S", ['random selection', 'roulette wheel'])        

        # The propability of doing crossover of two genomes, if 0 don't use crossover
        C = trial.suggest_categorical("C", [0, 0.1, 0.3, 0.5, 0.6, 0.8, 0.95])    
       
        # The number of slices for n-crossover, if 0 use uniform crossover
        if C>0:
            N = trial.suggest_categorical("N", [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
        else:
            N = None    

        # The propability of doing mutation on a bit of a genome, if 0 don't use mutation
        M = trial.suggest_categorical("M", [0, 0.01, 0.1, 0.33, 0.5])

    if problem == 19:
        # Size of the genome population
        P = trial.suggest_categorical("P", [10, 20, 50, 100, 150, 200, 250, 300])
        
        # Type of selection
        S = trial.suggest_categorical("S", ['random selection', 'roulette wheel'])        

        # The propability of doing crossover of two genomes, if 0 don't use crossover
        C = trial.suggest_categorical("C", [0, 0.5, 0.6, 0.8, 0.95])    
       
        # The number of slices for n-crossover, if 0 use uniform crossover
        if C>0: 
            N = trial.suggest_categorical("N", [0, 1, 2, 4, 6, 8])        
        else: 
            N = None               

        # The propability of doing mutation on a bit of a genome, if 0 don't use mutation
        M = trial.suggest_categorical("M", [0, 0.01, 0.1, 0.33, 0.5])         

    # Keep track of best fitness
    best_fitness = []
    best_genomes = []

    # Run repetitions
    F, _logger = create_problem(problem, dimension)
    for _ in tqdm(range(repetitions), desc="Loading..."):
    # for _ in range(repetitions): 
        model = GA(F, budget, dimension)
        model.setparameters(P, S, C, N, M)
        model.main()
        best_fitness.append(model.best_fitness)
        best_genomes.append(model.best_genome)
        F.reset()
    _logger.close()
    
    return np.average(best_fitness)
    
def tune(total_trials):
    print("--- Started Tuning ---")
    
    # Creating optuna study
    study_name = "GA-study"
    storage_name = "{}.csv".format(study_name)
    study = optuna.create_study(sampler=RandomSampler(), direction="maximize", study_name=study_name)
    
    # Optimize the objective function
    for _ in range(total_trials):
        study.optimize(objective, n_trials=1)

    # Save results of study
    results = study.trials_dataframe()
    results.to_csv(storage_name)
    
    return results
    
def results(problem):
    results = pandas.read_csv(f'GA{problem}-study.csv')    
    results = results.sort_values(by=['value'], ascending=False) 
    print(results[['number', 'value', 'params_C', 'params_M', 'params_N', 'params_P', 'params_S', 'state']].head(10))
    
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
    
def addresults(problem):
    newrows = pandas.read_csv('GA-study.csv', index_col='Unnamed: 0')
    results = pandas.read_csv(f'GA{problem}-study.csv', index_col='Unnamed: 0')
    newrows["number"] = newrows["number"] + len(results)
    concat_df = pandas.concat([results,newrows])
    concat_df = concat_df.reset_index(drop=True)
    concat_df.to_csv('GA{problem}-study.csv')
        
def main():
    results(19)
    # tune(480)
    # addresults(19)

if __name__ == "__main__":
    main()