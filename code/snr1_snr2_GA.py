import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass
from GeneticAlgorithm import GA

# To make your results reproducible (not required by the assignment), you could set the random seed by
# `np.random.seed(some integer, e.g., 42)`

def create_problem(fid: int, dimension: int):
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="genetic_algorithm",  # name of your algorithm
        algorithm_info="Practical assignment of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l

if __name__ == "__main__":
    
    budget = 5000
    dimension = 50
    repetitions = 20
    
    """
    BEST: 4.253071742631119
    P = 44
    S = 'roulette wheel'
    C = 0.5
    N = 10
    M = 0
    """
    
    np.random.seed(1)
    # Tuneable parameters
    P = 44                  # Size of the genome population
    S = 'roulette wheel'    # The choice between 'random selection' and 'roulette wheel'
    C = 0.5                 # The propability of doing crossover of two genomes, if 0 don't use crossover
    N = 10                  # The number of slices for n-crossover, if 0 use uniform crossover
    M = 0                   # The propability of doing mutation on a bit of a genome, if 0 don't use mutation
        
    best_fitness = []
    
    F18, _logger = create_problem(18, dimension)
    for rep in range(repetitions): 
        GA18 = GA(F18, budget, dimension)
        GA18.setparameters(P, S, C, N, M)
        GA18.main()
        best_fitness.append(GA18.best_fitness)
        F18.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder
                
    print(np.average(best_fitness))
    
    """
    BEST: 45.6
    P = 200
    S = 'roulette wheel'
    C = 0.2
    N = 4
    M = 0
    """
    
    np.random.seed(1)
    # Tuneable parameters
    P = 200                 # Size of the genome population
    S = 'roulette wheel'    # The choice between 'random selection' and 'roulette wheel'
    C = 0.2                 # The propability of doing crossover of two genomes, if 0 don't use crossover
    N = 4                   # The number of slices for n-crossover, if 0 use uniform crossover
    M = 0                   # The propability of doing mutation on a bit of a genome, if 0 don't use mutation
   
    best_fitness = []
    
    F19, _logger = create_problem(19, dimension)
    for rep in range(repetitions): 
        GA19 = GA(F19, budget, dimension)
        GA19.setparameters(P, S, C, N, M)
        GA19.main()
        best_fitness.append(GA19.best_fitness)
        F19.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder
    
    print(np.average(best_fitness))