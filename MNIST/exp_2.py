r"""Exp 3:
- The codes are used to plot Fig. 4 in the paper.
"""
from utils import get_args
from utils import main
from utils import EXP_DIR


args = get_args()
# assert args.noniid
assert not args.LT


LOG_DIR = EXP_DIR + "exp_2/"

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

    # Set font
    font = {"size": 11}
    plt.rc("font", **font)

    # Define color palette
    palette = {
        "FGC": "red",        # FGC in red
        "GM": "blue",       # GM in blue
        "CC": "green",      # CC in green
        "TM": "orange",     # TM in orange
        "CM": "purple",     # CM in purple
        "Average": "black",  # Set black for AVG
        "Krum": "cyan"      # KRUM in cyan
    }

    # Define aggregator order
    agg_order = ["Average", "GM", "TM", "CM", "Krum", "CC", "FGC"]

    def exp_grid():
        for agg in ["acp", "rfa", "tm", "cp", "cm", "krum", "avg", "nacp"]:
            for seed in [1, 2, 3]:
                for bucketing in [0]:
                    for momentum in [0.99]:
                        for attack in ["SF", "IPM"]:
                            for n in [15]:
                                for f in [3]:
                                    for niid in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                                        for lr in [0.05]:
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
                            "krum": "Krum",
                            "avg": "Average"
                        }.get(agg, agg.upper()),
                        "seed": seed,
                        "f": f,
                        "niid": niid,
                    }
                )
        except Exception as e:
            pass

    results = pd.DataFrame(results)

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    results.to_csv(OUT_DIR + "exp2.csv", index=None)

    # Compute average accuracy over the last 100 iterations
    last_100_iterations = results[results['Iterations'] > (MAX_BATCHES_PER_EPOCH * EPOCHS - 100)]

    
    average_results = last_100_iterations.groupby(['ATK', 'Aggregator', 'niid', 'seed'])['Accuracy (%)'].mean().reset_index()

    # Create subplots
    attacks = ['SF', 'IPM'] + [atk for atk in average_results['ATK'].unique() if atk not in ['SF', 'IPM']]
    num_attacks = len(attacks)
    fig, axes = plt.subplots(num_attacks, 1, figsize=(8.5, 5.5 * num_attacks), sharex=True)

    for ax, attack in zip(axes, attacks):
        attack_data = average_results[average_results['ATK'] == attack]
        for agg in agg_order:
            if agg in attack_data['Aggregator'].values:
                agg_data = attack_data[attack_data['Aggregator'] == agg]
                avg_accuracy = agg_data.groupby('niid')['Accuracy (%)'].mean().reset_index()
                
                # Use color palette
                color = palette.get(agg, "black")  # Default color is black
                ax.plot(avg_accuracy['niid'], avg_accuracy['Accuracy (%)'], marker='o', label=agg, color=color)

        ax.set_title(f'{attack}')
        ax.set_ylabel('Accuracy (%)')
        ax.legend(title='Aggregator', loc='center left')
        ax.grid(True)

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=1)  # Increase vertical spacing

    plt.xlabel(r'$\rho$')
    plt.tight_layout()
    plt.savefig(OUT_DIR + "accuracy_non-iid0.pdf", bbox_inches="tight", dpi=720)
    plt.show()