import pandas as pd
from datetime import datetime as dt
from tqdm import tqdm
from src.problem import instance_generator as ins
from src.problem.smpp import SMPP
from src.optimizers import (MINLPOptimizer,
                    GreedyHeuristicOptimizer,
                    GRASPOptimizer,
                    MINLPWarmStartOptimizer)


def main():
    # Max waiting time
    timeout=2*60
    # Verbose level
    verbose=1
    seed=42
    
    df_results = pd.DataFrame(columns=["optimizer_name", "N", "M", "d", "idx", 
                                    "time", "LB", "UB", "is_valid"])
    # instances = ins.list_avaliable_instances("instances/big*.json")[:1]
    # instances = ins.list_avaliable_instances("instances/small*50*10*0.2*0.json")
    # instances = ins.list_avaliable_instances("instances/small*.json")[:5]
    instances = ins.list_avaliable_instances("instances/small*50*25*0.8*3.json")
    print('Total of instances:', len(instances))
    """
            {
            'opt': MINLPOptimizer,
            'kwargs': {}
        },
    """
    opts = [

        {
            'opt': MINLPWarmStartOptimizer,
            'kwargs': {}
        }
    ]

    for name in tqdm(instances):
        if verbose: print(f'Running instance: {name}')
        # Load instance
        instance = ins.load(f'instances/{name}')
        for opt in opts:
            # Create optimizer and model instance
            optimizer = opt['opt']()
            smpp = SMPP(instance)

            # Run optimization
            result = optimizer.solve(smpp, timeout, seed, verbose, **opt['kwargs'])
            if verbose: print(result, '\n')

            vins = ins.get_info_from_name(name)
            df_results = df_results.append(
                {
                    "optimizer_name": result['name'],
                    "N": instance[0],
                    "M": instance[1],
                    "d": vins['d'],
                    "idx": vins['idx'],
                    "time": round(result['time'], 4),
                    "LB": round(result['LB'], 2),
                    "UB": round(result['UB'], 2),
                    "is_valid": result['is_valid']
                },
                ignore_index=True,
            )
    file_id = dt.now().isoformat().replace(':', '-').replace('.', '-')
    df_results.to_csv(f"results/{file_id}.csv", index=False)


if __name__ == "__main__":
    main()
