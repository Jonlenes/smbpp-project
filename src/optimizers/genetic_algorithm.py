import numpy as np
import random
from time import time
from .base import BaseOptimizer
from src.util import get_gurobi_model
from src.problem import SMBPP, Result
from gurobipy import GRB  

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