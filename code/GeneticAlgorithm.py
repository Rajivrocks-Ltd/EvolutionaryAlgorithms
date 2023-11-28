import numpy as np
from random import choices, random, sample
import time 

class GA():
    """"""
    def __init__(self, problem, budget, dimension, size):
        """"""
        self.problem = problem
        self.budget = budget
        self.dim = dimension
        self.pop_size = size
    
    def __creategenome(self) -> list:
        """"""
        genome = choices(population=[0, 1], k=self.dim)
        return genome
    
    def __evaluategenome(self, genome: list) -> float:
        """"""
        fitness = self.problem(genome)
        return fitness
        
    def __initialization(self):
        """"""
        self.pop = [self.__creategenome() for _ in range(self.pop_size)]
    
    def __mutation(self, genome: list, p: float) -> list:
        """"""
        if p < 0:
            raise ValueError(f"The value for p can not be negative! -> Choose a p between 0 and 1")
        elif p > 1:
            raise ValueError(f"The value for p is to big! -> Choose a p between 0 and 1")
        
        for idx in range(self.dim):
            if random() > p:
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
    
    def __selection(self, size: int) -> list:
        """"""
        if size > self.pop_size:
            raise ValueError(f"The selection size is higher than the size of the population! -> Size of a population: {self.pop_size}")
        elif size < 1:
            raise ValueError(f"Select at least 1 genome of the population!")
        
        fitness = [self.__evaluategenome(genome) for genome in self.pop]
        selection = choices(population=self.pop, weights=fitness, k=size) # arg 'weights': parameter to weigh the possibility for each value.
        return selection

    def __generation(self):
        """"""
        A = [1,1,1,1,1,1,1,1,1,1]
        B = [0,0,0,0,0,0,0,0,0,0]
        
        self.__initialization()
        self.__mutation(genome=A, p=0.5)
        self.__ncrossover(genome_A=A, genome_B=B, n=3)
        self.__unicrossover(genome_A=A, genome_B=B, p=0.5)
        self.__selection(size=3)

    def main(self):
        """"""
        # `problem.state.evaluations` counts the number of function evaluation automatically,
        # which is incremented by 1 whenever you call `problem(x)`.
        # You could also maintain a counter of function evaluations if you prefer.
        
        it = 1
        while self.problem.state.evaluations < self.budget:
            print(f"--- iteration {it} ---")
            
            # please call the mutation, crossover, selection here
            self.__generation()
            
            it += 1
            print(f"number of function evaluation: {self.problem.state.evaluations}\n")
