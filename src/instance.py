import os
import math
import json
import numpy as np
from glob import glob
from util import read_json, save_json

def generate_instance(n, m):
    # Orçamento do cliente
    b = np.random.randint(1000, 20000)
    # Produtos que farão parte do pacote do cliente
    P = np.random.randint(1, m, (np.random.randint(1, n)))
    S = np.zeros((m))
    S[P] = 1
    return [n, m, b, P, S]

def generate_and_save_all(save_folder='instances'):
    os.makedirs(save_folder, exist_ok=True)

    # Quantidade de produtos
    N = [10, 100, 200, 500, 1000]
    # Quantidade de clientes
    M = [100, 1000, 2000, 10000, 20000]

    for index, (n, m) in enumerate(zip(N, M)):
        ins = generate_instance(n, m)
        json_ins = instance2json(ins)
        save_json(os.path.join(save_folder, f'instancia-{index}.json'), json_ins)

def instance2json(ins):
    names = ['n', 'm', 'b', 'P', 'S']
    dic_ins = {}
    for name, value in zip(names, ins):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        dic_ins[name] = value
    return json.dumps(dic_ins)


def list_avaliable_instances(regex="instances/*.json"):
    return glob(regex)

def load(filename):
    json_ins = read_json(filename)
    return [
        json_ins["n"],
        json_ins["m"],
        json_ins["b"],
        json_ins["P"],
        json_ins["S"]
    ]

if __name__ == "__main__":
    generate_and_save_all()