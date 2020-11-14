import numpy as np
from gurobipy import quicksum

class SMPP:
    def __init__(self, instance):
        self.n_prodcuct, self.n_clients, self.clients = instance
        self._x = [False] * self.n_clients
        self._p = [0] * self.n_prodcuct

    def set_prices(self, p):
        self._p = p
    
    def get_current_prices(self):
        return self._p

    def set_client_decision(self, client_idx, buy):
       self._x[client_idx] = int(buy)

    def get_clients_decision(self):
        return self._x

    def get_objective_function(self):
        return SMPP.objective_function

    def sort_clients_by_budget(self):
        self.clients = sorted(self.clients, key=lambda k: k['b'], reverse=True)

    def current_cost(self):
        return SMPP.objective_function(self._p, self._x, self.clients)

    def validate_current_solution(self):
        return SMPP.validate(self._p, self._x, self.clients)

    def get_maximum_revenue(self):
        return sum(client['b'] for client in self.clients)

    @staticmethod
    def objective_function(p, x, clients):
        sum_function = sum if isinstance(p, list) and isinstance(x, list) else quicksum
        
        return sum_function(
                SMPP.cost_by_client(p, client, sum_function) * x[j]
                for j, client in enumerate(clients)
            )

    @staticmethod
    def cost_by_client(p, client, sum_function=sum):
        return sum_function(p[i] for i in client['S'])


    @staticmethod
    def constraints_gen(p, x, clients, only_who_bought=False):
        for j, client in enumerate(clients):
            if not only_who_bought or bool(x[j]):
                yield ((SMPP.cost_by_client(p, client, quicksum) - client['b']) * x[j] <= 0)

    @staticmethod
    def validate(p, x, clients):
        for j, client in enumerate(clients):
            cost = (SMPP.cost_by_client(p, client, sum) - client['b']) * int(x[j])
            if cost > 0 and not np.isclose(cost, 0):
                return False
        return True