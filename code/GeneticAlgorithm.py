import numpy as np
from random import choices, uniform, sample
import time 

class GA():
    """"""
    def __init__(self, problem, budget, dimension):
        """"""
        # Fixed Parameters
        self.problem = problem  # Problem to solve
        self.budget = budget    # The fixed budget, maximum number of evaluations of a individual
        self.dim = dimension    # Dimension of bit strings
        self.cash = {}          # Cash dictionary that keeps track of fitness scores of already evaluated genomes
        self.params = False     # Boolean that is False if parameters are not set and True otherwise
        self.best_fitness = 0
        
    def setparameters(self, size, S, Pc, N, Pm):
        """"""
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
        """"""
        if self.pop_size % 2 == 1:
            raise ValueError(f"The given population size is uneven! -> Choose an even number for the population size")

        if not isinstance(self.S, (bool)):
            raise ValueError(f"The value of S is not a boolean! -> Choose a S of True (Proportional Selection) or False (Random Selection)")

        if self.Pc < 0:
            raise ValueError(f"The value for pc can not be negative! -> Choose a pc between 0 and 1")
        elif self.Pc > 1:
            raise ValueError(f"The value for pc is to big! -> Choose a pc between 0 and 1")
        
        if self.N > self.dim-1:
            raise ValueError(f"Not enough dimension to have n splits! -> Number of dimension: {self.dim}")
        elif self.N < 0:
            raise ValueError(f"A negative number of splits is not possible!")

        if self.Pm < 0:
            raise ValueError(f"The value for pm can not be negative! -> Choose a pm between 0 and 1")
        elif self.Pm > 1:
            raise ValueError(f"The value for pm is to big! -> Choose a pm between 0 and 1")
    
    def genome2string(self, genome: list) -> str:
        """"""
        genomestr = ''.join(str(g) for g in genome)
        return genomestr
    
    def __creategenome(self) -> list:
        """"""
        genome = choices(population=[0, 1], k=self.dim)
        return genome
    
    def __evaluategenome(self, genome: list) -> float:
        """"""
        genomestr = self.genome2string(genome)
        if genomestr in self.cash:
            fitness = self.cash[genomestr]
        else:
            fitness = self.problem(genome)
            self.cash[genomestr] = fitness
        return fitness
    
    def __evaluategeneration(self, pop: list) -> list:
        """"""
        fitness = [self.__evaluategenome(genome) for genome in pop]   
        if max(fitness) > self.best_fitness:
                self.best_fitness = max(fitness)
        return fitness
        
    def __initialization(self) -> list:
        """"""
        pop = [self.__creategenome() for _ in range(self.pop_size)]
        return pop
        
    def __selection(self, pop: list, fitness: list) -> list:
        """"""        
        if self.S:
            selection = choices(population=pop, weights=fitness, k=self.pop_size) # arg 'weights': parameter to weigh the possibility for each value.
            # selection = self.__proportionalselection(pop=pop, fitness=fitness)
        else:
            selection = choices(population=pop, k=self.pop_size)
        
        return selection
            
    # def __proportionalselection(self, pop: list, fitness: list) -> list:
    #     """"""            
    #     total_fitness = sum(fitness) # Calculate the total fitness of the population
    #     proportional_fitness = [fit / total_fitness for fit in fitness] # Calculate the proportional fitness for each individual
    #     roulette_wheel = list(np.cumsum(proportional_fitness)) # Calculate the cumulative propabilities for the roulette    
        
    #     selected_individuals = []
    #     for _ in range(self.pop_size):
    #         selected_individuals.append(pop[next(index for index, value in enumerate(roulette_wheel) if uniform(0,1) < value)])
        
    #     return selected_individuals

    def __ncrossover(self, genome_A: list, genome_B: list) -> tuple:
        """"""        
        splits = sample(range(1, self.dim), k=self.N)
        splits.sort()
        for idx in splits:
            A=genome_A # make copy of genome A
            genome_A = genome_A[:idx] + genome_B[idx:] # perform crossover after index
            genome_B = genome_B[:idx] + A[idx:]
        return genome_A, genome_B
    
    def __unicrossover(self, genome_A: list, genome_B: list) -> tuple:
        """"""        
        for idx in range(self.dim):
            if uniform(0,1) > 0.5:
                A=genome_A[idx] # make copy of bit idx from genome A
                genome_A[idx] = genome_B[idx]
                genome_B[idx] = A
        return genome_A, genome_B

    def __mutation(self, genome: list) -> list:
        """"""        
        for idx in range(self.dim):
            if uniform(0,1) < self.Pm:
                genome[idx] = np.abs(genome[idx]-1)
        return genome
    
    def __newgeneration(self, pop: list, fitness: list) -> list:
        """"""
        # SELECTION
        pop = self.__selection(pop=pop, fitness=fitness)
        
        # CROSSOVER
        if self.Pc > 0: # Check if crossover is toggled on
            if self.N > 0: # If True: perform n-point crossover
                for idx in range(int(self.pop_size/2)):
                    if uniform(0,1) < self.Pc:
                        pop[2*idx], pop[2*idx+1] = self.__ncrossover(genome_A=pop[2*idx], genome_B=pop[2*idx+1])
            else: # If False: perform uniform crossover
                for idx in range(int(self.pop_size/2)):
                    if uniform(0,1) < self.Pc:
                        pop[2*idx], pop[2*idx+1] = self.__unicrossover(genome_A=pop[2*idx], genome_B=pop[2*idx+1])
                    
        # MUTATION
        if self.Pm > 0: # Check if mutation is toggled on
            for idx, genome in enumerate(pop):
                pop[idx] = self.__mutation(genome=genome)
        
        # EVALUATION
        fitness = self.__evaluategeneration(pop)
        
        return pop, fitness

    def main(self):
        """"""
        if self.params == False:
            raise InterruptedError("Please first set the parameters for the model with class.setparameters()!")
        
        pop = self.__initialization()
        fitness = self.__evaluategeneration(pop)
        gen = 1
        while self.problem.state.evaluations < self.budget:
            # print(f"--- generation {gen} ---")
            pop, fitness = self.__newgeneration(pop, fitness)     
            gen += 1
            if gen == 10000: break