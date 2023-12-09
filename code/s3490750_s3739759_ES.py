import numpy as np
# you need to install this package `ioh`. Please see documentations here:
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass

# TODO 1: Implement random search over the hyperparameters of the evolutionary strategy

budget = 5000
dimension = 50

# To make your results reproducible (not required by the assignment), you could set the random seed by
np.random.seed(69)


def set_hyper_parameters(problem_id: int):
    if problem_id == 18:
        # Set hyperparameters specific to problem F18
        return {
            "mu": 5,
            "lambda_": 100,
            "stagnation_threshold": 10,
            "threshold_divisor": 3,
            "initial_mutation_rate": 1.0,
            "mutation_increase": 1.4,
            "mutation_decay": 0.99
        }
    elif problem_id == 19:
        # Set hyperparameters specific to problem F19
        return {
            "mu": 20,
            "lambda_": 50,
            "stagnation_threshold": 50,
            "threshold_divisor": 10,
            "initial_mutation_rate": 1.0,
            "mutation_increase": 1.1,
            "mutation_decay": 0.99
        }


def initialize_cache():
    """Initialize an empty cache for storing fitness evaluations."""
    return {}


def get_individual_key(individual):
    """Create a unique key for each individual based on its genes."""
    return str(individual[0])  # Convert the numpy array of genes to a string


def evaluate_fitness(individual, problem, cache):
    """Evaluate the fitness of an individual, using cached values if available."""
    individual = individual.astype(int).tolist()  # Convert the numpy array to a list of ints
    key = get_individual_key(individual)
    if key in cache:
        # Retrieve the fitness from the cache
        fitness = cache[key]
    else:
    # Evaluate the fitness and update the cache
        if problem.state.evaluations >= budget:
            return -1
        fitness = problem(individual[0])
        cache[key] = fitness
    return fitness


def initialize_population(num_individuals):
    """Initialize the population with random genes and sigma values."""
    population = []
    for _ in range(num_individuals):
        genes = np.random.uniform(-1, 1, dimension)  # Random genes
        sigmas = np.random.uniform(0.1, 1, dimension)  # Random sigma values
        individual = np.array([genes, sigmas])
        population.append(individual)

    population = np.array(population)
    return population


def mutate(individual, problem, mutation_rate):
    """Mutate an individual using its sigma values."""
    genes, sigmas = individual
    tau_prime = 1 / np.sqrt(2 * problem.meta_data.n_variables)
    tau = 1 / np.sqrt(2 * np.sqrt(problem.meta_data.n_variables))
    g = np.random.normal(0, 1)  # Global random value for all sigmas

    # Mutate sigma values
    new_sigmas = sigmas * np.exp(tau_prime * g + tau * np.random.normal(0, 1, dimension))

    # Further adjusting the sigma values by the dynamic mutation rate
    adjusted_sigmas = new_sigmas * mutation_rate

    # Mutate genes
    new_genes = genes + adjusted_sigmas * np.random.normal(0, 1, dimension)
    return np.array([new_genes, new_sigmas])


def recombine(parent1, parent2):
    """Discrete recombination."""
    child_genes, child_sigmas = [], []
    for i in range(dimension):
        if np.random.rand() < 0.5:
            child_genes.append(parent1[0][i])
            child_sigmas.append(parent1[1][i])
        else:
            child_genes.append(parent2[0][i])
            child_sigmas.append(parent2[1][i])
    return np.array([np.array(child_genes), np.array(child_sigmas)])


def convert_to_binary_representation(population):
    """Convert the population's genes to a binary representation where genes below zero are zero and above zero are one."""
    binary_population = []
    for individual in population:
        binary_genes = np.where(individual[0] > 0, 1, 0).astype(int)
        # Ensure the individual structure is maintained as [genes, sigmas]
        binary_individual = np.array([binary_genes, individual[1]])
        binary_population.append(binary_individual)
    return np.array(binary_population)


# Modify the select function to use caching
def select(original_population, problem, cache, mu_):
    """Select the top mu individuals based on the fitness of modified copies."""
    fitness_evaluations = []
    binary_population = convert_to_binary_representation(original_population)

    for individual in binary_population:
        # The fitness is evaluated using the binary representation of the genes
        fitness = evaluate_fitness(individual, problem, cache)
        if fitness == -1:
            return -1
        fitness_evaluations.append(fitness)

    ranked_population = [ind for _, ind in
                         sorted(zip(fitness_evaluations, original_population), key=lambda x: x[0], reverse=True)]
    selected_population = ranked_population[:mu_]
    return selected_population


def s3490750_s3739759_ES(problem, run, fid):
    """The main function implementing the evolutionary strategy."""
    # Set hyperparameters
    params = set_hyper_parameters(fid)
    mu_ = params["mu"]
    lambda_ = params["lambda_"]
    stagnation_threshold = params["stagnation_threshold"]
    threshold_divisor = params["threshold_divisor"]
    initial_mutation_rate = params["initial_mutation_rate"]
    mutation_increase = params["mutation_increase"]
    mutation_decay = params["mutation_decay"]

    population = initialize_population(mu_)
    cache = initialize_cache()
    prev_evaluation_count = 0
    stagnation_count = 0
    mutation_rate = initial_mutation_rate # Initial mutation rate
    curr_run = run
    print(f'Run: {curr_run}')

    while problem.state.evaluations < budget:
        # Print run number if it changes
        if run is not None:
            if curr_run != run:
                curr_run = run
                print(f'Run: {run}')

        # Dynamically adjust mutation rate, sort of convergence velocity based on stagnation
        if stagnation_count >= stagnation_threshold // threshold_divisor:
            mutation_rate *= mutation_increase  # Increase mutation rate
        else:
            mutation_rate *= mutation_decay  # Decrease mutation rate

        # Create offspring by recombination and mutation of parents in the population and evaluate their fitness values
        offspring = []
        for _ in range(lambda_):
            parent1 = population[np.random.randint(mu_)]
            parent2 = population[np.random.randint(mu_)]
            new_offspring = mutate(recombine(parent1, parent2), problem, mutation_rate)
            offspring.append(new_offspring)

        population = select(offspring, problem, cache, mu_)
        if population == -1:
            # print(f'Run: {run} reached budget max budget. Stopping...')
            break

        # Check for stagnation
        current_evaluation_count = len(cache)
        if current_evaluation_count == prev_evaluation_count:
            stagnation_count += 1
        else:
            stagnation_count = 0
            mutation_rate = initial_mutation_rate
        prev_evaluation_count = current_evaluation_count

        # Restart if stagnation is detected
        if stagnation_count >= stagnation_threshold:
            print(f'Stagnation detected in run: {run} at {problem.state.evaluations} evaluations. Restarting...')
            cache = initialize_cache()  # Optionally, clear the cache
            population = initialize_population(mu_)  # Restart the population
            stagnation_count = 0
            mutation_rate = initial_mutation_rate


def random_search(problem):
    """Baseline random search algorithm."""
    while problem.state.evaluations < budget:
        solution = np.random.randint(0, 2, size=(dimension, dimension))
        problem(solution)
    return problem.state.current_best


def create_problem(fid: int, name: str, algorithm_name: str, directory: str):
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload
    # it to IOHanalyzer.

    l = logger.Analyzer(
        root=directory,
        folder_name=f'{name} run',  # the folder name to which the raw performance data will be stored
        algorithm_name=algorithm_name,  # name of your algorithm
        algorithm_info="Practical assignment of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    F18, _logger = create_problem(18, "F18", "ES", "ESData-f18")
    for run in range(20):
        s3490750_s3739759_ES(F18, run, 18)
        F18.reset()  # it is necessary to reset the problem after each independent run
    _logger.close()  # after all runs, it is necessary to close the logger to make sure all data are written to the disk

    F19, _logger = create_problem(19, "F19", "ES", "ESData-f19")
    for run in range(20):
        s3490750_s3739759_ES(F19, run, 19)
        F19.reset()
    _logger.close()

    # Run baseline random search algorithm to compare with ES
    # F18rs, _logger = create_problem(18, "F18RS", "RS", "ESData-f18-rs")
    # solutions = []
    # for run in range(20):
    #     solutions.append(random_search(F18rs))
    #     F18rs.reset()
    # _logger.close()
    #
    # # Run baseline random search algorithm to compare with ES
    # F19rs, _logger = create_problem(19, "F19RS", "RS", "ESData-f19-rs")
    # solutions = []
    # for run in range(20):
    #     solutions.append(random_search(F19rs))
    #     F19rs.reset()
    # _logger.close()
