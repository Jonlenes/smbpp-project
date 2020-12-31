from gurobipy import GRB
from .base import BaseOptimizer
from src.util import get_gurobi_model
from src.problem import SMBPP, Result

class MINLPOptimizer(BaseOptimizer):
    def __init__(self):
        self._x = None
        self._p = None

    def set_warm_start(self, x, p):
        self._x = x
        self._p = p

    def _solve(self, smbpp, timeout, seed, verbose, **kwargs):
        """
        Find which clients will be satisfied and the best prices using the non-linear model.
        """
        if verbose: print('MINLPOptimizer')
        model = get_gurobi_model(timeout, verbose)

        # Variables: Buy decision e Prices 
        clients_decision = model.addVars(smbpp.n_clients, vtype=GRB.BINARY, name="clients_decision")
        prices = model.addVars(smbpp.n_product, vtype=GRB.CONTINUOUS, name="prices")

        if self._x and self._p:
            if verbose: print('\tWarm-start is being used')
            for j in range(smbpp.n_clients):
                clients_decision[j].start = self._x[j]

            for i in range(smbpp.n_product):
                prices[i].start = self._p[i]

        # Set objective function
        model.setObjective(
            SMBPP.objective_function(prices, clients_decision, smbpp.clients),
            GRB.MAXIMIZE,
        )

        # Add constraints
        model.addConstrs(
            (c for c in SMBPP.constraints_gen(prices, clients_decision, smbpp.clients, False))
        )

        # Print stats
        if verbose == 2:
            model.printStats()

        # Solve the model
        model.optimize()
        smbpp.set_prices(model.getAttr('x', prices).values())
        smbpp.set_clients_decision(model.getAttr('x', clients_decision).values())

        result = Result()
        result['name'] = self.__class__.__name__
        result['LB'] = model.objVal
        result['UB'] = model.ObjBound
        return result