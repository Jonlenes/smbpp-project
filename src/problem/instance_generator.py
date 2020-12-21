import os
import math
import json
import random
import numpy as np
from glob import glob
from src.util import read_json, save_json

def generate_instance(n, m):
    clients = []
    # Para cada cliente
    for _ in range(m):
        # Orçamento do cliente
        b = np.random.randint(1000, 20000)
        # Produtos que farão parte do pacote do cliente
        P = random.sample(range(n), np.random.randint(1, n))
        clients.append({'b': b, 'S': P})
    return [n, m, clients]

def generate_and_save_all(save_folder='instances'):
    os.makedirs(save_folder, exist_ok=True)

    # Quantidade de produtos
    N = [10, 10, 10, 10, 100, 200, 500, 1000]
    # Quantidade de clientes
    M = [10, 30, 50, 100, 1000, 2000, 10000, 20000]

    for index, (n, m) in enumerate(zip(N, M)):
        ins = generate_instance(n, m)
        json_ins = instance2json(ins)
        save_json(os.path.join(save_folder, f'instance-{index}.json'), json_ins)

def instance2json(ins):
    names = ['n', 'm', 'clients']
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
        json_ins['n'],
        json_ins['m'],
        json_ins['clients'],
    ]

if __name__ == "__main__":
    generate_and_save_all()