from gurobipy import GRB
from .base import BaseOptimizer
from src.util import get_gurobi_model
from src.problem import SMBPP, Result
from .non_linear import MINLPOptimizer
from .greedy_heuristic import GreedyHeuristicOptimizer
from time import time

class MINLPWarmStartOptimizer(BaseOptimizer):
    def _solve(self, smbpp, timeout, seed, verbose, **kwargs):
        """
        .
        """
        # Create greedy optimizer
        if verbose: print('MINLPWarmStartOptimizer')
        start_time = time()
        greedy_opt = GreedyHeuristicOptimizer()
        result = greedy_opt.solve(smbpp, timeout, seed, verbose=0, **kwargs)
        if verbose:
            print('\tGreedy runned: %.4fs' % (time() - start_time))
            print('\tBest Objective Value: (%.2f, %.2f)' % (result['LB'], result['UB']))
        
        # Create non linear optimizers
        minlp_opt = MINLPOptimizer()
        minlp_opt.set_warm_start(smbpp.get_clients_decision(), smbpp.get_current_prices())
        result = minlp_opt.solve(smbpp, timeout, seed, verbose, **kwargs)

        result['name'] = self.__class__.__name__
        return result