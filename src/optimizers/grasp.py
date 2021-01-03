import random
from gurobipy import GRB
from .base import BaseOptimizer
from src.util import get_gurobi_model
from src.problem import SMBPP, Result
from time import time

class GRASPOptimizer(BaseOptimizer):
    def _solve(self, smbpp, timeout, seed, verbose, **kwargs):
        """
        Performs a Greedy Randomized Adaptive Search Procedure
        """

        if verbose: print('GRASPOptimizer')
        best_cost = grasp(smbpp, timeout, seed=seed, verbose=verbose, **kwargs)
        result = Result()
        result['name'] = self.__class__.__name__
        result['LB'] = best_cost
        result['UB'] = smbpp.get_maximum_revenue()
        return result

def grasp(smbpp, timeout, iterations, alpha, seed, verbose):
    start_time = time()
    random.seed(seed)
    best_S, best_cost = [], 0
    for i in range(iterations):
        S, cost = constructive_heuristic(smbpp, alpha, verbose)
        S, cost = local_search(smbpp, S, cost, verbose)
        if cost > best_cost:
            best_S, best_cost = S, cost
        if verbose:
            print(f"\tIter.: {i}, BestSol = {best_cost}")
        if time()-start_time > timeout:
            break
    return best_cost

def evaluate_candidates(smbpp, CL, S, current_cost):
    """
    Evaluate the incremental cost c(e) for all e in CL
    """
    # Set all client_decision (x) to zero
    smbpp.reset_current_solution()

    # Set all elements already in the solution
    for s in S:
        smbpp.set_client_decision(s, True)

    # Compute the incremental cost
    costs = {}
    for e in CL:
        smbpp.set_client_decision(e, True)
        # TODO: verificar se a solução já não atende o cliente e.
        # Tipo o que é feito no greedy: SMBPP.cost_by_client(smbpp.get_current_prices(), client) >= client['b']...
        cost, _ = optimize(smbpp, 0)
        costs[e] = cost - current_cost
        smbpp.set_client_decision(e, False)

    return costs   

def constructive_heuristic(smbpp, alpha, verbose = 0):
    # Create the candidate list
    CL = [j for j in range(smbpp.n_clients)]
    # Create empty RCL
    RCL = []
    # Start with empty solution
    current_cost = 0
    S = []

    iter = 0
    while CL:
        if verbose == 2:
            print("\t\tConstructive heuristic iteration: ", iter, " Current cost: ", current_cost, 
                " CL length: ", len(CL))
        # Evaluate the incremental cost c(e) for all e in CL
        costs = evaluate_candidates(smbpp, CL, S, current_cost)

        # Compute cost min and max
        c_min = min(costs.values())
        c_max = max(costs.values())

        # Build RCL
        for e in CL:
            if costs[e] >= c_min + alpha * (c_max - c_min):
                RCL.append(e)

        # Will stop when we have no element in the RCL or no improve
        if c_min + alpha * (c_max - c_min) < 0 or not RCL: break
        
        # Select an element s from the RCL at random
        s = random.choice(RCL)
        RCL = []
        # Add s to the solution
        S += [s]
        current_cost += costs[s]
        # Update candidate set
        CL.remove(s)
        iter += 1

    return S, current_cost

def local_search(smbpp, best_sol, cost, verbose):
    # Initialize the solution
    smbpp.reset_current_solution()
    for s in best_sol:
        smbpp.set_client_decision(s, True)

    # Stores the clients that are out of the solution
    in_candidates = []
    for cli, dec in enumerate(smbpp.get_clients_decision()):
        if dec == 0:
            in_candidates.append(cli)

    best_cost = -1
    
    iter = 0
    while best_cost < cost:
        best_cost = cost
        in_cand, out_cand = None, None
        if verbose == 2:
            print("\t\tLocal search iteration: ", iter, " Best cost: ", best_cost)

        #Explore the neighborhoods
        cost, in_cand = add_neighborhood(smbpp, best_sol, best_cost, in_candidates)
        if best_cost >= cost:
            cost, out_cand = remove_neighborhood(smbpp, best_sol, best_cost)
        if best_cost >= cost:
            cost, in_cand, out_cand = exchange_neighborhood(smbpp, best_sol, best_cost, in_candidates)
        
        #Perform the changes in the solution
        if out_cand is not None:
            smbpp.set_client_decision(out_cand, False)
            best_sol.remove(out_cand)
        if in_cand is not None:
            smbpp.set_client_decision(in_cand, True)
            best_sol.append(in_cand)
        iter += 1

    return best_sol, best_cost

def add_neighborhood(smbpp, S, cost, in_candidates):
    """
    Performs a first improving search adding a new client in the solution
    """ 
    for cand in in_candidates:
        smbpp.set_client_decision(cand, True)
        new_cost, _ = optimize(smbpp, 0)
        if new_cost > cost:
            return new_cost, cand
        smbpp.set_client_decision(cand, False)
    return cost, None

def remove_neighborhood(smbpp, S, cost):
    """
    Performs a first improving search removing a client from the solution
    """
    for cand in S:
        smbpp.set_client_decision(cand, False)
        new_cost, _ = optimize(smbpp, 0)
        if new_cost > cost:
            return new_cost, cand
        smbpp.set_client_decision(cand, True)
    return cost, None

def exchange_neighborhood(smbpp, S, cost, in_candidates):
    """
    Performs a first improving search exchanging a client in the solutio by a client out of the solution
    """
    for in_cand in in_candidates:
        for out_cand in S:
            smbpp.set_client_decision(in_cand, True)
            smbpp.set_client_decision(out_cand, False)
            new_cost, _ = optimize(smbpp, 0)
            if new_cost > cost:
                return new_cost, in_cand, out_cand
            smbpp.set_client_decision(out_cand, True)
        smbpp.set_client_decision(in_cand, False)
    return cost, None, None



def optimize(smbpp: SMBPP, verbose: int):
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