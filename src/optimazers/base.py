from abc import ABC, abstractmethod
from time import time
from src.problem import Result

class BaseOptimiser(ABC):
    """Abstract class for Optimisers"""

    def __init__(self):
        pass

    def solve(self, smpp, timeout=None, verbose=0) -> Result:
        """Abstract method to optimize.
        """
        start_time = time()
        result = self._solve(smpp, timeout=None, verbose=0)
        result['time'] = time() - start_time
        result['is_valid'] = smpp.validate_current_solution()
        return result
    
    @abstractmethod
    def _solve(self, smpp, timeout, verbose):
        """Abstract method to optimize.
        """
        raise NotImplementedError