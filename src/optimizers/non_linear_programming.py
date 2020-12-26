from gurobipy import GRB
from .base import BaseOptimizer
from src.util import get_gurobi_model
from src.problem import SMPP, Result

class MINLPOptimizer(BaseOptimizer):
    def _solve(self, smpp, timeout, verbose):
        """
        Find which clients will be satisfied and the best prices using the non-linear model.
        """
        model = get_gurobi_model(timeout, verbose)

        # Variables: Buy decision e Prices 
        clients_decision = model.addVars(smpp.n_clients, vtype=GRB.BINARY, name="clients_decision")
        prices = model.addVars(smpp.n_prodcuct, vtype=GRB.CONTINUOUS, name="prices")

        # Set objective function
        model.setObjective(
            SMPP.objective_function(prices, clients_decision, smpp.clients),
            GRB.MAXIMIZE,
        )

        # Add constraints
        model.addConstrs(
            (c for c in SMPP.constraints_gen(prices, clients_decision, smpp.clients, False))
        )

        # Print stats
        if verbose:
            model.printStats()

        # Solve the model
        model.optimize()

        result = Result()
        result['name'] = self.__class__.__name__
        result['LB'] = model.objVal
        result['UB'] = model.ObjBound
        return result