r"""Exp 3:
- The codes are used for Table in the paper.
"""
from utils import get_args
from utils import main
from utils import EXP_DIR
import numpy as np

args = get_args()
assert args.noniid
assert not args.LT


LOG_DIR = EXP_DIR + "exp_8/"

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
    EPOCHS = 50

if not args.plot:
    main(args, LOG_DIR, EPOCHS, MAX_BATCHES_PER_EPOCH)
else:
    # Temporarily put the import functions here to avoid
    # random error stops the running processes.
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from codes.parser import extract_validation_entries

    # 5.5in is the text width of iclr2022 and 11 is the font size
    font = {"size": 11}
    plt.rc("font", **font)

    def exp_grid():
        for agg in [ "acp", "rfa", "tm", "cp", "cm", "krum", "avg", "nacp" ]:
            for seed in [1, 2, 3]:
                for bucketing in [0]:
                    for momentum in [0.99]:
                        for attack in ["GA", "SF", "LF", "mimic", "IPM", "ALIE"]:
                            for n in [15]:
                                for f in [6]:
                                    for niid in [0.5]:
                                        for lr in [0.02]:
                                            for b in [32]:
                                                yield niid, b, lr, agg, attack, n, f, momentum, bucketing, seed

    results = []
    for niid, b, lr, agg, attack, n, f, momentum, bucketing, seed in exp_grid():
        grid_identifier = f"niid{niid}_{lr}_{agg}_{attack}_n{n}_f{f}_{momentum}_s{bucketing}_seed{seed}"
        path = INP_DIR + grid_identifier + "/stats"
        try:
            values = extract_validation_entries(path)
            for v in values:
                results.append(
                    {
                        "Iterations": v["E"] * MAX_BATCHES_PER_EPOCH,
                        "Accuracy (%)": v["top1"],
                        "niid": niid,
                        "ATK": attack,
                        "Aggregator": {
                            "nacp": "FGC",
                            "rfa": "GM",
                            "cp": "CC",
                            "tm": "TM",
                            "cm": "CM",
                            "avg": "Average",
                            "krum": "Krum"
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

    results.to_csv(OUT_DIR + "exp2.csv", index=None)

    last_100_iterations = results[results['Iterations'] > (MAX_BATCHES_PER_EPOCH * EPOCHS - 100)]

    
    average_results = last_100_iterations.groupby(['ATK', 'Aggregator', 'seed'])['Accuracy (%)'].mean().reset_index()

   
    final_results = average_results.groupby(['ATK', 'Aggregator']).agg({'Accuracy (%)': ['mean', 'std']}).reset_index()
    final_results.columns = ['ATK', 'Aggregator', 'Accuracy (%)', 'Std Dev']

    
    print(final_results)
