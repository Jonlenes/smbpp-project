import pandas as pd
from datetime import datetime as dt
from tqdm import tqdm
from src.problem import instance_generator as ins
from src.problem.smpp import SMPP
from src.optimizers import MINLPOptimizer, GreedyHeuristicOptimizer


def main():
    # Max waiting time
    timeout=10000*60
    # Verbose level
    verbose=0
    
    df_results = pd.DataFrame(columns=["optimizer_name", "N", "M", "time", "LB", "UB"])
    instances = ins.list_avaliable_instances("instances/small*.json")
    print('Total of instances:', len(instances))

    for name in tqdm(instances):
        if verbose: print(f'Running instance: {name}')
        # Load instance
        instance = ins.load(f'instances/{name}')
        for optimizer in [GreedyHeuristicOptimizer(), MINLPOptimizer()]:
            smpp = SMPP(instance)
            result = optimizer.solve(smpp, timeout, verbose)
            if verbose: print(result, '\n')

            df_results = df_results.append(
                {
                    "optimizer_name": result['name'],
                    "N": instance[0],
                    "M": instance[1],
                    "time": round(result['time'], 2),
                    "LB": round(result['LB'], 2),
                    "UB": round(result['UB'], 2)
                },
                ignore_index=True,
            )
    file_id = dt.now().isoformat().replace(':', '-').replace('.', '-')
    df_results.to_csv(f"results/{file_id}.csv", index=False)


if __name__ == "__main__":
    main()
