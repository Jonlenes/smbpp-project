import json
import gurobipy as gp

def get_gurobi_model(timeout=None, verbose=0, seed=42, name='smbpp'):
    # Create a new model
    model = gp.Model(name)

    # Set Params
    model.setParam(gp.GRB.Param.OutputFlag, int(verbose==2))
    model.setParam(gp.GRB.Param.Seed, seed)

    if timeout:
        model.setParam(gp.GRB.Param.TimeLimit, timeout)

    return model

def read_json(filename):
    with open(filename, "r") as file:
        data = json.loads(file.read())
    return data

def save_json(filename, str_json):
    with open(filename, 'w') as file:
        file.write(str_json)