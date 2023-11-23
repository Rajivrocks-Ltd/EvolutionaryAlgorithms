class GA():

    def __init__(self, budget):
        self.budget = budget
        pass
    
    def initialization(self):
        """Initialization of the population
        
        Args:
            arg1 (_type_): _description_
            arg2 (_type_): _description_
        """
        pass
        # _IMPLEMENT:_ initial_pop = ... make sure you randomly create the first population
    
    def mutation(self):
        """_summary_ 
        
        Args:
            arg1 (_type_): _description_
            arg2 (_type_): _description_
        """
        pass
    
    def crossover(self):
        """_summary_
        
        Args:
            arg1 (_type_): _description_
            arg2 (_type_): _description_
        """        
        pass
    
    def selection(self):
        """_summary_
        
        Args:
            arg1 (_type_): _description_
            arg2 (_type_): _description_
        """
        pass
    
    def run(self, problem):
        """_summary_

        Args:
            problem (_type_): _description_
        """
        # `problem.state.evaluations` counts the number of function evaluation automatically,
        # which is incremented by 1 whenever you call `problem(x)`.
        # You could also maintain a counter of function evaluations if you prefer.
        while problem.state.evaluations < self.budget:
            pass
            # please call the mutation, crossover, selection here
            # f = problem(x): this is how you evaluate one solution `x`