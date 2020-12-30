from gurobipy import GRB
from .base import BaseOptimizer
from src.util import get_gurobi_model
from src.problem import SMBPP, Result

class GreedyHeuristicOptimizer(BaseOptimizer):
    def _solve(self, smbpp, timeout, seed, verbose, **kwargs):
        """
        Performs a greedy heuristic that is based on adding the solution to the client 
        with the largest possible budget.
        """
        best_cost = 0.0
        # Sort client by their budgets
        smbpp.sort_clients_by_budget()
        # For each client
        for j, client in enumerate(smbpp.clients):
            # Add client to the solution
            smbpp.set_client_decision(j, True)
            # Check if it is the first client or if the current prioce does not satisfy the 
            # purchase for the current client
            if j == 0 or SMBPP.cost_by_client(smbpp.get_current_prices(), client) >= client['b']:
                # Get new prrices
                current_cost, prices = optimize(smbpp)
                # Check if the solution improve
                if current_cost > best_cost:
                    best_cost = current_cost
                    smbpp.set_prices(prices)
                else:
                    smbpp.set_client_decision(j, False)
            else:
                # This is great, keep the prices and compute the new revenue
                best_cost = smbpp.current_cost()

        smbpp.set_prices()
        result = Result()
        result['name'] = self.__class__.__name__
        result['LB'] = best_cost
        result['UB'] = smbpp.get_maximum_revenue()
        return result


def optimize(smbpp: SMBPP, verbose: int=0):
    """
    Given the clients that must be satisfied, it computes the best prices.
    """
    model = get_gurobi_model(verbose=verbose)

    # Variables: Prices
    prices = model.addVars(smbpp.n_prodcuct, vtype=GRB.CONTINUOUS, name="prices")

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