from gurobipy import GRB
from .base import BaseOptimizer
from src.util import get_gurobi_model
from src.problem import SMPP, Result
from .non_linear import MINLPOptimizer
from .greedy_heuristic import GreedyHeuristicOptimizer
from time import time

class MINLPWarmStartOptimizer(BaseOptimizer):
    def _solve(self, smpp, timeout, seed, verbose, **kwargs):
        """
        .
        """
        # Create greedy optimizer
        start_time = time()
        greedy_opt = GreedyHeuristicOptimizer()
        result = greedy_opt.solve(smpp, timeout, seed, verbose, **kwargs)
        if verbose:
            print('\tGreedy runned: %.4fs' % (time() - start_time))
            print('\tBest Objective Value: (%.2f, %.2f)' % (result['LB'], result['UB']))
            print(smpp.get_clients_decision())
            print(smpp.get_current_prices())
        
        # Create non linear optimizers
        minlp_opt = MINLPOptimizer()
        minlp_opt.set_warm_start(smpp.get_clients_decision(), smpp.get_current_prices())
        result = minlp_opt.solve(smpp, timeout, seed, verbose, **kwargs)

        result['name'] = self.__class__.__name__
        return result