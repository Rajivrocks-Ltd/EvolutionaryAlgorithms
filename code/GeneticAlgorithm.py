import numpy as np
from numpy.random import choice, uniform, sample

class GA():
    """Class for a Genetic Algorithm and all its functionalities"""
    
    def __init__(self, problem, budget, dimension):
        """Initialize all fixed parameters for the created GA"""
        # Fixed Parameters
        self.problem = problem  # Problem to solve
        self.budget = budget    # The fixed budget, maximum number of evaluations of a individual
        self.dim = dimension    # Dimension of bit strings
        self.cache = {}         # Cache dictionary that keeps track of fitness scores of already evaluated genomes
        self.cached = 0         # Integer that keeps count of the consecutive times the evaluation used the cache 
        self.params = False     # Boolean that is False if parameters are not set and True otherwise
        self.best_fitness = 0   # Keep track of the best fitness off all generations
        self.best_genome = None # Keep track of the genome with the best fitness off all generations
        
    def setparameters(self, size, S, Pc, N, Pm):
        """Function to set all the tuneable parameters"""
        # Tuneable parameters
        self.pop_size = size    # Size of the genome population
        self.S = S              # If proportional selections should be used instead of random selection
        self.Pc = Pc            # The propability of doing crossover of two genomes, if 0 don't use crossover
        self.N = N              # The number of slices for n-crossover, if 0 use uniform crossover
        self.Pm = Pm            # The propability of doing mutation on a bit of a genome, if 0 don't use mutation
        
        # Checker for tuneable parameters
        self.__error_checker()
        
        # Parameters are set
        self.params = True
        
    def __error_checker(self) -> None:
        """Function to check if the values for the parameters are correctly set, to prevent errors"""
        if self.pop_size % 2 == 1:
            raise ValueError(f"The given population size is uneven! -> Choose an even number for the population size")

        if self.S not in ['random selection', 'roulette wheel']:
            raise ValueError(f"This value of S is not possible! -> Choose a S of 'random selection' or 'roulette wheel'")

        if self.Pc < 0:
            raise ValueError(f"The value for pc can not be negative! -> Choose a pc between 0 and 1")
        elif self.Pc > 1:
            raise ValueError(f"The value for pc is to big! -> Choose a pc between 0 and 1")
        
        if self.N == None:
            pass
        elif self.N > self.dim-1:
            raise ValueError(f"Not enough dimension to have n splits! -> Number of dimension: {self.dim}")
        elif self.N < 0:
            raise ValueError(f"A negative number of splits is not possible!")

        if self.Pm < 0:
            raise ValueError(f"The value for pm can not be negative! -> Choose a pm between 0 and 1")
        elif self.Pm > 1:
            raise ValueError(f"The value for pm is to big! -> Choose a pm between 0 and 1")
    
    def __genome2string(self, genome: list) -> str:
        """Function to convert list of genome bit to a string of bits"""
        genomestr = ''.join(str(g) for g in genome)
        
        return genomestr
    
    def __creategenome(self) -> list:
        """Function to create genomes as a list of 0s and/or 1s"""
        genome = [choice([0, 1]) for _ in range(self.dim)]
        
        return genome
    
    def __evaluategenome(self, genome: list) -> float:
        """Function to evaluate a genome with the caching capability"""
        # Convert genome to string
        genomestr = self.__genome2string(genome)
        
        # Check if genome was already in cache, if so fetch its fitness
        if genomestr in self.cache:
            fitness = self.cache[genomestr]
            self.cached += 1
            
        # If not, add to cache and calculate fitness
        else:
            fitness = self.problem(genome)
            self.cache[genomestr] = fitness
            self.cached = 0
        
        return fitness
    
    def __evaluategeneration(self, pop: list) -> list:
        """Function to evaluate a population of a generation"""
        
        # Obtain a list of all fitnesses from all genomes of the population
        fitness = [self.__evaluategenome(genome) for genome in pop]
        
        # Check if new genome is obtained with the best fitness
        if max(fitness) > self.best_fitness:
            self.best_fitness = max(fitness)
            self.best_genome = pop[fitness.index(max(fitness))]
            
        return fitness
        
    def __initialization(self) -> list:
        """Function to initialize the population of the GA"""
        pop = [self.__creategenome() for _ in range(self.pop_size)]
        
        return pop
        
    def __selection(self, pop: list, fitness: list) -> list:
        """Function to perform random selection or roulette selection"""   
        
        # Check if roulette wheel should be performed
        if self.S == 'roulette wheel':
            selection = self.__roulettewheel(pop=pop, fitness=fitness)
            
        # Otherwise perform random selection
        else:
            selection = self.__randomselection(pop=pop)
            
        return selection
            
    def __randomselection(self, pop: list) -> list:
        """Function to perform random selection"""
        
        # List of selected individuals and loop until pop_size number of individuals are selected
        selected_individuals = []
        for _ in range(self.pop_size):
            # Randomly select index of a genome in the population and add the genome to the list
            idx = choice(np.arange(0, self.pop_size))
            selected_individuals.append(pop[idx])
            
        return selected_individuals        
        
    def __roulettewheel(self, pop: list, fitness: list) -> list:
        """Function to perform roulette wheel selection"""            
        
        # Calculate the cumulative propabilities for the roulette with the proportional fitnesses for each genome
        total_fitness = sum(fitness)
        proportional_fitness = [fit / total_fitness for fit in fitness]
        roulette_wheel = list(np.cumsum(proportional_fitness))
        
         # List of selected individuals and loop until pop_size number of individuals are selected
        selected_individuals = []
        for _ in range(self.pop_size):
            # Spin the roulette, select index of a genome in the population and add the genome to the list
            spin = uniform(0,1)
            selected_index = 0
            while roulette_wheel[selected_index] < spin:
                selected_index += 1
            selected_individuals.append(pop[selected_index])
        
        return selected_individuals

    def __ncrossover(self, genome_A: list, genome_B: list) -> tuple:
        """Function to perform n-point crossover"""
        
        # Select the indices where the genomes need to split
        splits = choice(np.arange(1, self.dim), size=self.N)
        splits.sort()
        
        # Split the genome and flip the parts of the parents
        for idx in splits:
            A=genome_A
            genome_A = genome_A[:idx] + genome_B[idx:]
            genome_B = genome_B[:idx] + A[idx:]
            
        return genome_A, genome_B
    
    def __unicrossover(self, genome_A: list, genome_B: list) -> tuple:
        """Function to perform uniform crossover"""    
        
        # Loop over all bits of the two genomes and flip the bits of the parents based on uniform distribution 
        for idx in range(self.dim):
            if uniform(0,1) > 0.5:
                A=genome_A[idx]
                genome_A[idx] = genome_B[idx]
                genome_B[idx] = A
                
        return genome_A, genome_B

    def __mutation(self, genome: list) -> list:
        """Function to perform bit-wise mutation"""      
        # Loop over all bits of the two genomes and flip bit to its opposite based on uniform distribution   
        for idx in range(self.dim):
            if uniform(0,1) < self.Pm:
                genome[idx] = np.abs(genome[idx]-1)
                
        return genome
    
    def __newgeneration(self, pop: list, fitness: list) -> list:
        """Function to perform all operators of the GA, this includes the stagnation emergency case"""
                
        # EMERGENCY CASE WHEN STAGNATION
        if (self.cached > self.pop_size*3) and (self.Pm == 0):
            self.cached = 0
            self.Pm = 0.1
            for idx, genome in enumerate(pop):
                self.__mutation(genome=genome)
            self.Pm = 0
        
        # SELECTION
        pop = self.__selection(pop=pop, fitness=fitness)
        
        # CROSSOVER
        if self.Pc > 0:
            if self.N > 0:
                for idx in range(int(self.pop_size/2)):
                    if uniform(0,1) < self.Pc:
                        pop[2*idx], pop[2*idx+1] = self.__ncrossover(genome_A=pop[2*idx], genome_B=pop[2*idx+1])
            else:
                for idx in range(int(self.pop_size/2)):
                    if uniform(0,1) < self.Pc:
                        pop[2*idx], pop[2*idx+1] = self.__unicrossover(genome_A=pop[2*idx], genome_B=pop[2*idx+1])
        
        # MUTATION
        if self.Pm > 0:
            for idx, genome in enumerate(pop):
                pop[idx] = self.__mutation(genome=genome)
        
        # EVALUATION
        fitness = self.__evaluategeneration(pop)

        return pop, fitness

    def main(self):
        """Function to perform training of the Genetic Algorithm"""
        
        # Check if tuneable parameters are set
        if self.params == False:
            raise InterruptedError("Please first set the parameters for the model with class.setparameters()!")
        
        # initialize population and determine its fitness
        pop = self.__initialization()
        fitness = self.__evaluategeneration(pop)
        
        # Generate new generations until budget is met
        gen = 1
        while self.problem.state.evaluations < self.budget:
            pop, fitness = self.__newgeneration(pop, fitness)     
            gen += 1