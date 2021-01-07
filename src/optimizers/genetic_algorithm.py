import numpy as np
import random
from time import time
from .base import BaseOptimizer
from src.util import get_gurobi_model
from src.problem import SMBPP, Result
from gurobipy import GRB         
        

class GAOptimizer(BaseOptimizer):

    def _solve(self, smbpp, timeout, seed, verbose, **kwargs):
        """
        Performs a Genetic Algorithm heuristic.
        """

        if verbose: print('GAOptimizer')
        best_cost = self.evolve(smbpp, timeout, seed=seed, verbose=verbose, **kwargs)
        result = Result()
        result['name'] = self.__class__.__name__
        result['LB'] = best_cost
        result['UB'] = smbpp.get_maximum_revenue()
        return result
                    
                    
    def _initialize_pop(self):
        """
        Generates the initial population.
        """
        for i in range(self.pop_size):
            self.pop[i].initialize()
        
    def _one_crossover(self, parent1, parent2, child1, child2, point):
        """
        Peforms the 1-point crossover.

        ### Parameters:
            :parent1, parent2: parents.
            :child1, child2: Chromosome objects to stores the results.
            :point: crossover point.

        :return: None
        """
        child1.solution[0:point] = parent1.solution[0:point]
        child1.solution[point:self.smbpp.n_clients] = parent2.solution[point:self.smbpp.n_clients]
        child1.fitness_value = None
        
        child2.solution[0:point] = parent2.solution[0:point]
        child2.solution[point:self.smbpp.n_clients] = parent1.solution[point:self.smbpp.n_clients]
        child2.fitness_value = None
        
    
    def _uniform_crossover(self, parent1, parent2, child1, child2):
        """
        Peforms the 1-point crossover.

        ### Parameters:
            :parent1, parent2: parents.
            :child1, child2: Chromosome objects to stores the results.  

        :return: None
        """

        for i in range(self.smbpp.n_clients):
            if random.random() < 0.5:
                child1.solution[i], child2.solution[i] = parent1.solution[i], parent2.solution[i]
            else:
                child1.solution[i], child2.solution[i] = parent2.solution[i], parent1.solution[i]
        child1.fitness_value = None
        child2.fitness_value = None
        
    def _mutation(self, cromo):
        """
        Performs the mutation

        :return: None
        """
        for i in range(self.smbpp.n_clients):
            if(random.random() <= self.mut_rate):
                cromo.solution[i] = 1-cromo.solution[i]
                
    
    def _calculate_fitness(self, beg, end):
        """
        Calculates the fitness of all chromosomes in the interval [beg,end].
        
        :return: None
        """
        for i in range(beg, end+1):
            if self.pop[i].fitness_value is None: # Computes the fitness only for new chromosomes
                self.pop[i].fitness(self.smbpp)
    
                
    def _roulette_wheel_selection(self):
        """
        Selects self.pop_size parents using the roulette wheel method.
        The indexes of the selected parents are stored in self.parents.
        
        :return: None
        """
        sum_fitness = 0
        for i in range(self.pop_size):
            self.probs[i] = self.pop[i].fitness_value
            sum_fitness += self.probs[i]

        # Computes the probability of each chromose based on his fitness
        for i in range(self.pop_size):
            self.probs[i] /= sum_fitness
            
        # Selects the parents based on the probabilities
        self.parents = np.random.choice(self.pop_size, self.pop_size, p=self.probs)
        
    def _offspring_generation(self):
        """
        Generates self.pop_size children using the crossover and the mutation.
        
        :return: None
        """
        for i in range(0, self.pop_size, 2):
            point = random.randint(1, self.pop_size-1)
            parent1, parent2 = self.parents[i], self.parents[i+1]
            if self.uniform_cross:
                self._uniform_crossover(self.pop[parent1], self.pop[parent2], self.pop[i+self.pop_size], 
                    self.pop[i+1+self.pop_size])
            else:
                self._one_crossover(self.pop[parent1], self.pop[parent2], self.pop[i+self.pop_size], 
                    self.pop[i+1+self.pop_size], point)
            self._mutation(self.pop[i+self.pop_size])
            self._mutation(self.pop[i+1+self.pop_size])
        
       
    def evolve(self, smbpp, timeout, seed, verbose, num_generations, pop_size, mut_rate, selection_method = 1, uniform_cross = True):
        """
        Runs the genetic algorithm.

        ### Parameters:
            :smbpp: problem instance.
            :timeout: time limit of execution.   
            :seed: random number generator seed.
            :verbose: (0 - Silence, 1 - Only infos, 2 - Debug).
            :num_generations: maximum number of generations.
            :pop_size: number of chromosomes of the population.
            :mut_rate: mutation rate.
            :selection_method: (0 - Roullete Wheel, 1 - Stochastic Universal Sampling, 2 - Tournament)
            :uniform_cross: (True - Uniform Crossover, False - One-Point Crossover).

        :return: None
        """
        self.smbpp = smbpp
        self.timeout = timeout
        self.pop = [Chromosome(self.smbpp.n_clients) for _ in range(2*pop_size)] #Populacao (a 2a metade da lista armazena os filhos)
        self.mut_rate = mut_rate
        self.parents = [None]*pop_size #Armazena os indices dos pais para reproducao
        self.probs = [None]*pop_size #Armazena as probabilidades para o metodo da roleta
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.uniform_cross = uniform_cross
        self.verbose = verbose
        self.selection_method = selection_method
        random.seed(seed)
        np.random.seed(seed)
        
        gen = 0 #Generation index        
        start_time = time()
        best_sol = -1 #Cost of the incumbent solution
        best_time = 0 #Time when the incumbent solution was found
        n_no_improvements = 0 #Number of consecutive generations without improvement
        
        self._initialize_pop()
        while(gen < self.num_generations and time() - start_time < self.timeout):
            self._calculate_fitness(0, self.pop_size-1) #Calcula o fitness dos pais (quando necessario)
            

            self._roulette_wheel_selection()
            
            self._offspring_generation() # Crossover and mutation
            self._calculate_fitness(self.pop_size, 2*self.pop_size-1)
            # Sorts the chromosomes so that the 1st half of self.pop (the new generation)
            # contains the best chromosomes
            self.pop.sort(reverse=True, key = lambda x: x.fitness_value) 
            
            n_no_improvements += 1
            # New incument solution
            if self.pop[0].fitness_value > best_sol:
                best_sol = self.pop[0].fitness_value
                best_time = time()-start_time
                n_no_improvements = 0
            
            # Re-initialize 10 chromosomes if no improve was found in the last 10 generations
            if n_no_improvements == 10:
                print("\n\tRestarting 10 chromosomes\n")
                indexes = np.random.choice(np.arange(20, 50), 10, replace = False)
                for i in indexes:
                    self.pop[i].initialize()
                n_no_improvements = 0
            
            # Prints the best fitness
            if(self.verbose == 1 and gen % 50 == 0 or self.verbose == 2):
                print("\n\tGeneration ", gen, " Best Fitness: ", self.pop[0].fitness_value)
                if verbose == 2:
                    print("\tTop 10:")
                    for i in range(10):
                        print("\t", self.pop[i].fitness_value)
                    print("\tBest time", best_time)
            
            gen += 1
        print('Best time: ', best_time)
        return self.pop[0].fitness_value


class Chromosome:
    
    def __init__(self, n_clients):
        
        self.solution = [None]*n_clients # Clients decisions
        self.fitness_value = None # Stores the value of the objective function
        
    def initialize(self):
        """
        Initializes the chromosome with random values.
        """
        for i in range(len(self.solution)):
            self.solution[i] = random.randint(0,1)
                        
    def fitness(self, smbpp):
        """
        Calculates the chromosome fitness.
        """
        smbpp.set_clients_decision(self.solution)
        self.fitness_value, _ = self._optimize(smbpp, 0)
        return self.fitness_value

    def _optimize(self, smbpp: SMBPP, verbose: int):
        """
        Given the clients that must be satisfied, it computes the best prices.
        """
        model = get_gurobi_model(verbose=verbose)

        # Variables: Prices
        prices = model.addVars(smbpp.n_product, vtype=GRB.CONTINUOUS, name="prices")

        # Set objective function
        model.setObjective(
            SMBPP.objective_function(prices, smbpp.get_clients_decision(), smbpp.clients),
            GRB.MAXIMIZE
        )

        # Add constraints
        model.addConstrs(
            (c for c in SMBPP.constraints_gen(prices, smbpp.get_clients_decision(), smbpp.clients, True))
        )

        # Solve the model
        model.optimize()
        
        return model.objVal, model.getAttr('x', prices).values()