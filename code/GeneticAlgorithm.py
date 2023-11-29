import numpy as np
from random import choices, random, sample
import time 

class GA():
    """"""
    def __init__(self, problem, dimension, size):
        """"""
        self.problem = problem
        self.dim = dimension
        self.pop_size = size
    
    def __creategenome(self) -> list:
        """"""
        genome = choices(population=[0, 1], k=self.dim)
        return genome
    
    def __evaluategenome(self, genome: list) -> float: # !Note: Calculates F(x) not E(x).
        """"""
        fitness = self.problem(genome)
        return fitness
    
    def __evaluategeneration(self, pop) -> list:
        """"""
        fitness = [self.__evaluategenome(genome) for genome in pop]        
        return fitness
        
    def __initialization(self) -> list:
        """"""
        pop = [self.__creategenome() for _ in range(self.pop_size)]
        return pop
        
    def __selection(self, pop: list, fitness: list, size: int) -> list:
        """"""
        if size > self.pop_size:
            raise ValueError(f"The selection size is higher than the size of the population! -> Size of a population: {self.pop_size}")
        elif size < 1:
            raise ValueError(f"Select at least 1 genome of the population!")
        
        selection = choices(population=pop, weights=fitness, k=size) # arg 'weights': parameter to weigh the possibility for each value.
        return selection
    
    def __mutation(self, genome: list, p: float) -> list:
        """"""
        if p < 0:
            raise ValueError(f"The value for p can not be negative! -> Choose a p between 0 and 1")
        elif p > 1:
            raise ValueError(f"The value for p is to big! -> Choose a p between 0 and 1")
        
        for idx in range(self.dim):
            if random() < p:
                genome[idx] = np.abs(genome[idx]-1)
                
        return genome
    
    def __ncrossover(self, genome_A: list, genome_B: list, n: int) -> tuple: # n-point cross over
        """"""
        if n > self.dim-1:
            raise ValueError(f"Not enough dimension to have n splits! -> Number of dimension: {self.dim}")
        elif n < 0:
            raise ValueError(f"A negative number of splits is not possible!")

        
        splits = sample(range(1, self.dim), k=n)
        splits.sort()
        
        for idx in splits:
            A=genome_A                               # make copy of genome A
            genome_A = genome_A[:idx] + genome_B[idx:]  # perform crossover after index
            genome_B = genome_B[:idx] + A[idx:]

        return genome_A, genome_B
    
    def __unicrossover(self, genome_A: list, genome_B: list, p: float) -> tuple: # uniform-point cross over
        """"""
        if p < 0:
            raise ValueError(f"The value for p can not be negative! -> Choose a p between 0 and 1")
        elif p > 1:
            raise ValueError(f"The value for p is to big! -> Choose a p between 0 and 1")
        
        A = []
        B = []
        for idx in range(self.dim):
            if random() > p:
                A.append(genome_A[idx])
                B.append(genome_B[idx])
            else:
                A.append(genome_B[idx])
                B.append(genome_A[idx])
        
        return A, B

    def __newgeneration(self, pop: list) -> list:
        """"""
        fitness = self.__evaluategeneration(pop)
        newpop = []
        for _ in range(int(self.pop_size/2)):
        
            # SELECTION
            S = 2
            parents = self.__selection(pop=pop, fitness=fitness, size=S)
            par1, par2 = parents[0], parents[1]
        
            # CROSSOVER
            N = 2
            pop[0], pop[1] = self.__ncrossover(genome_A=par1, genome_B=par2, n=N)
            # PU = 0.5
            # pop[0], pop[1] = self.__unicrossover(genome_A=par1, genome_B=par2, p=PU)
            newpop.extend((pop[0], pop[1]))
        
        # MUTATION
        for idx, genome in enumerate(newpop):
            PM = 0.1
            pop[idx] = self.__mutation(genome=genome, p=PM)
        
        return newpop

    def main(self, budget):
        """"""
        self.budget = budget
        pop = self.__initialization()
        gen = 1
        while self.problem.state.evaluations < self.budget:
            print(f"--- generation {gen} ---")
            
            # please call the mutation, crossover, selection here
            pop = self.__newgeneration(pop)
            
            gen += 1
            print(f"number of function evaluation: {self.problem.state.evaluations}\n")

        # evaluate final generation
        fitness = self.__evaluategeneration(pop)
        bestfitness = max(fitness)
        bestgenome = pop[fitness.index(bestfitness)]
        
        print(f"--- results of last generation: ---")
        print(fitness)
        print(pop)
        print(bestgenome, bestfitness)