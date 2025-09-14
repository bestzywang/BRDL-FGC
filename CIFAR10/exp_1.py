r"""Exp:
- The following codes are used for table iv in this paper
"""
from utils import get_args
from utils import main
from utils import EXP_DIR


args = get_args()
assert args.noniid
assert not args.LT


LOG_DIR = EXP_DIR + "exp_1/"

if args.identifier:
    LOG_DIR += f"{args.identifier}/"
elif args.debug:
    LOG_DIR += "debug/"
else:
    LOG_DIR += f"n{args.n}_f{args.f}_{args.noniid}/"

INP_DIR = LOG_DIR
OUT_DIR = LOG_DIR + "output33/"
LOG_DIR += f"niid{args.niid}_b{args.b}_{args.lr}_{args.agg}_{args.attack}_n{args.n}_f{args.f}_{args.momentum}_s{args.bucketing}_seed{args.seed}"

if args.debug:
    MAX_BATCHES_PER_EPOCH = 30
    EPOCHS = 3
else:
    MAX_BATCHES_PER_EPOCH = 30
    EPOCHS = 100

if not args.plot:
    main(args, LOG_DIR, EPOCHS, MAX_BATCHES_PER_EPOCH)
else:
    # Temporarily put the import functions here to avoid
    # random error stops the running processes.
    import json
    import os
    import numpy as np
    import pandas as pd
    from codes.parser import extract_validation_entries

    def exp_grid():
        for agg in [ "avg", "acp", "rfa", "tm", "cp", "cm", "krum", "nacp" ]:
            for seed in [1, 2, 3]:
                for bucketing in [0, 1, 2]:
                    for momentum in [0.9]:
                        for attack in ["GA", "BF", "SF", "LF", "mimic", "IPM", "ALIE"]:
                            for n in [25]:
                                for f in [5]:
                                    for niid in [0.3]:
                                        for lr in [0.25]:
                                            for b in [32]:
                                                yield niid, b, lr, agg, attack, n, f, momentum, bucketing, seed

    results = []
    for niid, b, lr, agg, attack, n, f, momentum, bucketing, seed in exp_grid():
        grid_identifier = f"niid{niid}_b{b}_{lr}_{agg}_{attack}_n{n}_f{f}_{momentum}_s{bucketing}_seed{seed}"
        path = INP_DIR + grid_identifier + "/stats"
        try:
            values = extract_validation_entries(path)
            for v in values:
                results.append(
                    {
                        "Iterations": v["E"] * MAX_BATCHES_PER_EPOCH,
                        "Accuracy (%)": v["top1"],
                        "ATK": attack,
                        "Aggregator": {
                            "nacp": "FGC",
                            "rfa": "GM",
                            "cp": "CC",
                            "tm": "TM",
                            "cm": "CM",
                            "krum": "KRUM"
                        }.get(agg, agg.upper()),
                        "seed": seed,
                        "f": f,
                        # r"s": bucketing,
                    }
                )
        except Exception as e:
            pass

    results = pd.DataFrame(results)
    print(results)
    # print(results.columns)
    # import ipdb; ipdb.set_trace()
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    results.to_csv(OUT_DIR + "exp3.csv", index=None)



    last_100_iterations = results[results['Iterations'] // MAX_BATCHES_PER_EPOCH > 90]

    # Compute average accuracy per attack and aggregator for each seed
    average_results = last_100_iterations.groupby(['ATK', 'Aggregator', 'seed'])['Accuracy (%)'].mean().reset_index()

    # For each aggregator and attack, compute mean and std over seeds 1, 2, 3
    final_results = average_results.groupby(['ATK', 'Aggregator']).agg({'Accuracy (%)': ['mean', 'std']}).reset_index()
    final_results.columns = ['ATK', 'Aggregator', 'Accuracy (%)', 'Std Dev']

    # Print tabular results
    print(final_results)
