from gurobipy import GRB
from .base import BaseOptimizer
from src.util import get_gurobi_model
from src.problem import SMPP, Result

class GreedyHeuristicOptimizer(BaseOptimizer):
    def _solve(self, smpp, timeout, seed, verbose, **kwargs):
        """
        Performs a greedy heuristic that is based on adding the solution to the client 
        with the largest possible budget.
        """
        best_cost = 0.0
        # Sort client by their budgets
        smpp.sort_clients_by_budget()
        # For each client
        for j, client in enumerate(smpp.clients):
            # Add client to the solution
            smpp.set_client_decision(j, True)
            # Check if it is the first client or if the current prioce does not satisfy the 
            # purchase for the current client
            if j == 0 or SMPP.cost_by_client(smpp.get_current_prices(), client) >= client['b']:
                # Get new prrices
                current_cost, prices = optimize(smpp)
                # Check if the solution improve
                if current_cost > best_cost:
                    best_cost = current_cost
                    smpp.set_prices(prices)
                else:
                    smpp.set_client_decision(j, False)
            else:
                # This is great, keep the prices and compute the new revenue
                best_cost = smpp.current_cost()

        result = Result()
        result['name'] = self.__class__.__name__
        result['LB'] = best_cost
        result['UB'] = smpp.get_maximum_revenue()
        return result


def optimize(smpp: SMPP, verbose: int=0):
    """
    Given the clients that must be satisfied, it computes the best prices.
    """
    model = get_gurobi_model(verbose=verbose)

    # Variables: Prices
    prices = model.addVars(smpp.n_prodcuct, vtype=GRB.CONTINUOUS, name="prices")

    # Set objective function
    model.setObjective(
        SMPP.objective_function(prices, smpp.get_clients_decision(), smpp.clients),
        GRB.MAXIMIZE
    )

    # Add constraints
    model.addConstrs(
        (c for c in SMPP.constraints_gen(prices, smpp.get_clients_decision(), smpp.clients, True))
    )

    # Solve the model
    model.optimize()
    
    return model.objVal, model.getAttr('x', prices).values()