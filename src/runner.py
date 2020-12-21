import pandas as pd


from tqdm import tqdm
from src.problem import instance_generator as ins
from src.problem.smpp import SMPP
from src.optimazers import MINLPOptimiser, GreedyHeuristicOptimiser


def main():
    # Choose: {0, 1, 2, 3, 4}
    instance_id = 0
    # Max waiting time
    timeout = 10000*60
    # Verbose level
    verbose = 0

    # Load instance
    instance = ins.load(f"instances/instance-{instance_id}.json")

    df_results = pd.DataFrame(columns=["optimazer_name", "N", "M", "time", "LB", "UB"])
    for optimiser in [GreedyHeuristicOptimiser(), MINLPOptimiser()]:
        smpp = SMPP(instance)
        result = optimiser.solve(smpp, timeout, verbose)
        
        if verbose:
            print(result)

        df_results = df_results.append(
            {
                "optimazer_name": result['name'],
                "N": instance[0],
                "M": instance[1],
                "time": round(result['time'], 2),
                "LB": round(result['LB'], 2),
                "UB": round(result['UB'], 2)
            },
            ignore_index=True,
        )
    df_results.to_csv(f"results/results_{instance_id}.csv", index=False)


if __name__ == "__main__":
    main()
