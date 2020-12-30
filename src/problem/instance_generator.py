import os
import re
import math
import json
import random
import numpy as np
from glob import glob
from itertools import product
from src.util import read_json, save_json

def decision(probability):
    return random.random() < probability

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

def generate_instance_paper(n, m, d):
    products_used = np.zeros(n, dtype=np.bool)
    clients, empty_bundles = [], []

    # For each client
    for j in range(m):
        # Clients budget
        b = np.random.randint(1, 1000)
        # Creating client bundle
        S = np.where([decision(d) for i in range(n)])[0].tolist()
        # Set product as used
        products_used[S] = True
        # Add client to the list
        clients.append({'b': b, 'S': S})
        if len(S) == 0:
            empty_bundles.append(j)
    
    # For each client with empty bundle, or each product not appearing in any
    # bundle, sample 1 product and client respectively and set Sji = 1
    unused_products = list(np.where(~products_used)[0])
    for j in empty_bundles:
        i = unused_products.pop() if len(unused_products) > 0 else np.random.randint(0, n)
        clients[j]['S'].append(int(i))
    for i in unused_products:
        j = np.random.randint(0, m)
        clients[j]['S'].append(int(i))
    return [n, m, clients]

def generate_and_save_all(save_folder='instances'):
    os.makedirs(save_folder, exist_ok=True)

    # Small set
    N = {10, 25, 50}           # Number of products
    M = {10, 25, 50, 100, 150} # Number of clients
    d = {0.2, 0.5, 0.8}        # Density of matrix S
    combinations = list(product(['small'], N, M, d))

    # Big set
    N = {100, 500}      # Number of products
    M = {150, 200, 300} # Number of clients
    d = {0.5}           # Density of matrix S
    combinations = combinations + list(product(['big'], N, M, d))

    for size, n, m, d in combinations:
        for idx in range(10):  # Generatin 10 instaces for each config
            ins = generate_instance_paper(n, m, d)
            json_ins = instance2json(ins)
            save_json(os.path.join(save_folder, f'{size}-N{n}M{m}d{d}-{idx}.json'), json_ins)

def instance2json(ins):
    names = ['n', 'm', 'clients']
    dic_ins = {}
    for name, value in zip(names, ins):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        dic_ins[name] = value
    return json.dumps(dic_ins)

def get_info_from_name(name):
    match = re.compile(r'N(?P<N>\d+)M(?P<M>\d+)d(?P<d>\d+\.\d+)-(?P<idx>\d+)').search(name)
    return {'N': int(match.group('N')),
        'M': int(match.group('M')),
        'd': float(match.group('d')),
        'idx': int(match.group('idx'))
    }

def list_avaliable_instances(regex="instances/*.json"):
    names = [path.split(os.path.sep)[-1] for path in glob(regex)]
    def cmp(name):
        reward = 1000 if 'big' in name else 0
        vins = get_info_from_name(name)
        return (reward + vins['N'] + vins['M'] + vins['d']*10 + vins['idx'])
    return sorted(names, key=cmp)

def load(filename):
    json_ins = read_json(filename)
    return [
        json_ins['n'],
        json_ins['m'],
        json_ins['clients'],
    ]

if __name__ == "__main__":
    generate_and_save_all()