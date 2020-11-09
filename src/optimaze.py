import os
import sys
import numpy as np
import gurobipy as gp
import pandas as pd
import instance as ins

from gurobipy import GRB
from itertools import combinations
from time import time
from tqdm import tqdm


def build_smpp_model(instance, timeout, verbose=0):
    """
    Build the SMPP model
    """
    # Create a new model
    model = gp.Model("smpp")

    # Set Params
    model.setParam(gp.GRB.Param.OutputFlag, verbose)
    model.setParam(gp.GRB.Param.TimeLimit, timeout)
    model.setParam(gp.GRB.Param.Seed, 42)

    # Intance
    n_prodcuct, n_clients, clients = instance

    # Variables: Buy decision e Prices 
    x = model.addVars(n_clients, vtype=GRB.BINARY, name="x")
    p = model.addVars(n_prodcuct, vtype=GRB.CONTINUOUS, name="p")

    # import pdb; pdb.set_trace();
    # Set objective
    model.setObjective(
        gp.quicksum(
            gp.quicksum(p[i] for i in client['S']) * x[j]
            for j, client in enumerate(clients)
        ),
        GRB.MAXIMIZE,
    )

    for j, client in enumerate(clients):
        model.addConstr(
            (gp.quicksum(p[i] for i in client['S']) - client['b']) * x[j] <= 0
        )

    if verbose:
        model.printStats()

    return model


def main():
    # Choose: {0, 1, 2, 3, 4}
    instance_id = 0
    # Max waiting time
    timeout = 10000*60
    # Verbose level
    verbose = 1

    # Load instance
    instance = ins.load(f"instances/instance-{instance_id}.json")

    # Building the models
    model = build_smpp_model(instance, timeout, verbose)
    model.optimize()

    import pdb; pdb.set_trace()
    pd.DataFrame(
        {
            "N": instance[0],
            "M": instance[1],
            "time": round(model.runtime, 2),
            "cost": round(model.objVal, 2)
        },
        index=[0],
    ).to_csv(f"results/results_{instance_id}.csv", index=False)


if __name__ == "__main__":
    main()
