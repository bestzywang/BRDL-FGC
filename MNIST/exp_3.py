r"""Exp 3:
- The codes are used to plot Fig. 5 in the paper.
"""
from utils import get_args
from utils import main
from utils import EXP_DIR


args = get_args()
# assert not args.noniid
assert not args.LT


LOG_DIR = EXP_DIR + "exp_3/"

if args.identifier:
    LOG_DIR += f"{args.identifier}/"
elif args.debug:
    LOG_DIR += "debug/"
else:
    LOG_DIR += f"n{args.n}_f{args.f}_{args.noniid}/"

INP_DIR = LOG_DIR
OUT_DIR = LOG_DIR + "output31/"
LOG_DIR += f"niid{args.niid}_{args.lr}_{args.agg}_{args.attack}_n{args.n}_f{args.f}_{args.momentum}_s{args.bucketing}_seed{args.seed}"

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

    # Update palette to assign different colors for each attack type
    palette = {
        "GA": "red",
        "LF": "blue",
        "mimic": "green",
        "ALIE": "orange",
        "SF": "purple",
        "IPM": "cyan"
    }
    # Custom attack order
    attack_order = ["GA", "LF", "mimic", "ALIE", "SF", "IPM"]
    
    def exp_grid():
        for agg in ["nacp"]:
            for seed in [1, 2, 3]:
                for bucketing in [0]:
                    for momentum in [0.99]:
                        for attack in ["GA", "LF", "mimic", "ALIE", "SF", "IPM"]:
                            for n in [15]:
                                for f in [1, 2, 3, 4, 5, 6, 7]:
                                    for niid in [0.1]:
                                        for lr in [0.02]:
                                            yield niid, lr, agg, attack, n, f, momentum, bucketing, seed

                                    

    results = []
    for niid, lr, agg, attack, n, f, momentum, bucketing, seed in exp_grid():
        grid_identifier = f"niid{niid}_{lr}_{agg}_{attack}_n{n}_f{f}_{momentum}_s{bucketing}_seed{seed}"
        path = INP_DIR + grid_identifier + "/stats"
        try:
            values = extract_validation_entries(path)
            for v in values:
                results.append(
                    {
                        "Iterations": v["E"] * MAX_BATCHES_PER_EPOCH,
                        "Accuracy (%)": v["top1"],
                        "ATK": attack,
                        "Aggregator": "FGC",  # Aggregator fixed to FGC since only nacp is used
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

    last_100_iterations = results[results['Iterations'] > (MAX_BATCHES_PER_EPOCH * EPOCHS - 10)]
    # Compute average accuracy per attack and aggregator per seed
    average_results = last_100_iterations.groupby(['ATK', 'f', 'seed'])['Accuracy (%)'].mean().reset_index()

    # Compute mean accuracy for each attack and f
    # average_results = results.groupby(['ATK', 'f', 'seed'])['Accuracy (%)'].mean().reset_index()

    # Compute overall mean accuracy for each attack and f
    final_average_results = average_results.groupby(['ATK', 'f'])['Accuracy (%)'].mean().reset_index()

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8.5, 6))

    ax.set_prop_cycle(color=[palette.get(attack, "black") for attack in attack_order])
    # Plot average accuracy for each attack type
    for attack in attack_order:
        attack_data = final_average_results[final_average_results['ATK'] == attack]
        ax.plot(attack_data['f'], attack_data['Accuracy (%)'], marker='o', label=attack)

    # Set legend
    ax.legend(title='Attack', loc='center left')

    # Set labels
    # ax.set_title('Average Accuracy by Number of Attacks')
    ax.set_xlabel(r'Number of Byzantine workers $f$')
    ax.set_ylabel(r'Accuracy ($\%$)')
    ax.set_ylim(0, 100)  # Set y-axis range
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
    ax.grid(True)

    # Show figure
    plt.tight_layout()
    plt.savefig(OUT_DIR + "combined_accuracy_by_f.pdf", bbox_inches="tight", dpi=720)
    plt.show()