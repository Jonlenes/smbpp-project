import random
from gurobipy import GRB
from .base import BaseOptimizer
from src.util import get_gurobi_model
from src.problem import SMPP, Result

class GRASPOptimizer(BaseOptimizer):
    def _solve(self, smpp, timeout, seed, verbose, **kwargs):
        """
        Performs ...
        """
        best_cost = grasp(smpp, seed=seed, verbose=verbose, **kwargs)
        result = Result()
        result['name'] = self.__class__.__name__
        result['LB'] = best_cost
        result['UB'] = smpp.get_maximum_revenue()
        return result

def grasp(smpp, iterations, alpha, seed, verbose):
    random.seed(seed)
    best_S, best_cost = [], 0
    for i in range(iterations):
        S, cost = constructive_heuristic(smpp, alpha)
        S, cost = local_search(smpp, S, cost)
        if cost > best_cost:
            best_S, best_cost = S, cost
            if verbose:
                print(f"\tIter.: {i}, BestSol = {best_cost}")
    return best_cost

def evaluate_candidates(smpp, CL, S, current_cost):
    """
    Evaluate the incremental cost c(e) for all e in CL
    """
    # Set all client_decision (x) to zero
    smpp.reset_current_solution()

    # Set all elements already in the solution
    for s in S:
        smpp.set_client_decision(s, True)

    # Compute the incremental cost
    costs = {}
    for e in CL:
        smpp.set_client_decision(e, True)
        # TODO: verificar se a solução já não atende o cliente e.
        # Tipo o que é feito no greedy: SMPP.cost_by_client(smpp.get_current_prices(), client) >= client['b']...
        cost, _ = optimize(smpp)
        costs[e] = cost - current_cost
        smpp.set_client_decision(e, False)

    return costs   

def constructive_heuristic(smpp, alpha):
    # Create the candidate list
    CL = [j for j in range(smpp.n_clients)]
    # Create empty RCL
    RCL = []
    # Start with empty solution
    current_cost = 0
    S = []

    while CL:
        # Evaluate the incremental cost c(e) for all e in CL
        costs = evaluate_candidates(smpp, CL, S, current_cost)

        # Compute cost min and max
        c_min = min(costs.values())
        c_max = max(costs.values())

        # Build RCL
        for e in CL:
            if costs[e] >= c_min + alpha * (c_max - c_min):
                RCL.append(e)

        # Will stop when we have no element in the RCL
        if not RCL: break
        
        # Select an element s from the RCL at random
        s = random.choice(RCL)
        RCL = []
        # Add s to the solution
        S += [s]
        current_cost += costs[s]
        # Update candidate set
        CL.remove(s)

    return S, current_cost

def local_search(smpp, S, cost):
    return S, cost

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