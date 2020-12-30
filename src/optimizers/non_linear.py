from gurobipy import GRB
from .base import BaseOptimizer
from src.util import get_gurobi_model
from src.problem import SMBPP, Result

class MINLPOptimizer(BaseOptimizer):
    def __init__(self):
        self._x = 0.0
        self._p = 0.0

    def set_warm_start(self, x, p):
        self._x = x
        self._p = p

    def _solve(self, smbpp, timeout, seed, verbose, **kwargs):
        """
        Find which clients will be satisfied and the best prices using the non-linear model.
        """
        model = get_gurobi_model(timeout, verbose)
        print(self._x)
        print(self._p)
        # Variables: Buy decision e Prices 
        clients_decision = model.addVars(smbpp.n_clients, vtype=GRB.BINARY, obj=self._x, name="clients_decision")
        prices = model.addVars(smbpp.n_prodcuct, vtype=GRB.CONTINUOUS, obj=self._p, name="prices")

        import pdb; pdb.set_trace()
        print(model.getAttr('x', prices).values())
        print(model.getAttr('x', clients_decision).values())

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
        if verbose:
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