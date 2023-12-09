import numpy as np
from tqdm import tqdm
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
    
    # Fixed parameters
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
    
    # # Tuneable parameters
    # S = 'roulette wheel'    # The choice between 'random selection' and 'roulette wheel'
    # M = 0                   # The propability of doing mutation on a bit of a genome, if 0 don't use mutation
        
    # best_f = 0
    # best_P = None
    # best_C = None
    # best_N = None
    # for P in [20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60]:
    #     for C in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #         for N in [1,2,4,6,8,10,12,14]:
                
    #             # Random seed for reproducibility
    #             np.random.seed(1)
                    
    #             # Running x number of repetitions and keep track of the obtained fitness scores
    #             best_fitness = []
    #             F18, _logger = create_problem(18, dimension)
    #             for _ in tqdm(range(repetitions), desc="Loading..."):   
    #                 GA18 = GA(F18, budget, dimension)
    #                 GA18.setparameters(P, S, C, N, M)
    #                 GA18.main()
    #                 best_fitness.append(GA18.best_fitness)
    #                 F18.reset()
    #             _logger.close()
                
    #             if np.average(best_fitness) > best_f:
    #                 best_f = np.average(best_fitness)
    #                 best_P = P
    #                 best_C = C
    #                 best_N = N
                            
    #             # Print average fitness after all repetitions
    #             print(P, C, N, np.average(best_fitness), f"  ->  best fitness {best_f}, P {best_P}, C {best_C}, N {best_N}")
    
    """
    BEST: 45.6
    P = 200
    S = 'roulette wheel'
    C = 0.2
    N = 4
    M = 0
    """
    
    # Tuneable parameters
    S = 'roulette wheel'    # The choice between 'random selection' and 'roulette wheel'
    M = 0                   # The propability of doing mutation on a bit of a genome, if 0 don't use mutation
    
    best_f = 0
    best_P = None
    best_C = None
    best_N = None
    for P in [20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,220,240,260,280,300]:
        for C in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for N in [1,2,4,6,8,10]:
              
                # Random seed for reproducibility
                np.random.seed(1)
                
                # Running x number of repetitions and keep track of the obtained fitness scores
                best_fitness = []
                F19, _logger = create_problem(19, dimension)
                for _ in tqdm(range(repetitions), desc="Loading..."):
                    GA19 = GA(F19, budget, dimension)
                    GA19.setparameters(P, S, C, N, M)
                    GA19.main()
                    best_fitness.append(GA19.best_fitness)
                    F19.reset() 
                _logger.close()
                
                if np.average(best_fitness) > best_f:
                    best_f = np.average(best_fitness)
                    best_P = P
                    best_C = C
                    best_N = N
                            
                # Print average fitness after all repetitions
                print(P, C, N, np.average(best_fitness), f"  ->  best fitness {best_f}, P {best_P}, C {best_C}, N {best_N}")
    