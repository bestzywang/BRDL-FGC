r"""Exp 3:
- The codes are used to plot Fig. 8 (a) in the paper.
"""
from utils import get_args
from utils import main
from utils import EXP_DIR


args = get_args()
assert not args.noniid
assert not args.LT


LOG_DIR = EXP_DIR + "exp_6/"

if args.identifier:
    LOG_DIR += f"{args.identifier}/"
elif args.debug:
    LOG_DIR += "debug/"
else:
    LOG_DIR += f"n{args.n}_f{args.f}_{args.noniid}/"

INP_DIR = LOG_DIR
OUT_DIR = LOG_DIR + "output1/"
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
    font = {"size": 10}
    plt.rc("font", **font)

    def exp_grid():
        for agg in ["nacp","nacpt"]:
            for seed in [1, 2, 3]:
                for bucketing in [0]:
                    for momentum in [0.99]:
                        for attack in ["GA", "LF", "mimic", "ALIE", "SF", "IPM"]:
                            for n in [15]:
                                for f in [ 2,  4, 6]:
                                    for niid in [0.3]:
                                        for lr in [0.05]:
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
                        "Attack": attack,
                        "Aggregator": {
                            "nacp": "FGC",
                            "nacpt": "FGC$_t$",
                        }.get(agg, agg.upper()),
                        "seed": seed,
                        "f": f,
                        r"s": bucketing,
                    }
                )
        except Exception as e:
            pass

    results = pd.DataFrame(results)

    # Convert lowercase attack names for display
    results['Attack'] = results['Attack'].str.replace('mimic', 'Mimic')

    print(results)
    # print(results.columns)
    # import ipdb; ipdb.set_trace()
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    results.to_csv(OUT_DIR + "exp2.csv", index=None)

    # Set color mapping
    palette = {
        "FGC": "red",        # FGC in red
        "FGC$_t$": "blue",        # FGC$_t$ in blue
    }


    # Compute average accuracy over the last 100 iterations
    last_100_iterations = results[results['Iterations'] > (MAX_BATCHES_PER_EPOCH * EPOCHS - 100)]

    # Compute mean, min, max accuracy for each attack and aggregator
    stats_results = last_100_iterations.groupby(['Attack', 'Aggregator', 'f', 's'])['Accuracy (%)'].agg(['mean', 'min', 'max']).reset_index()

    # Keep only required attacks and aggregators
    attacks = ['GA', 'LF', 'Mimic', 'ALIE', 'SF', 'IPM']
    agg_map = {'FGC': '-', 'FGC$_t$': '--'}

    # Filter data
    plot_data = stats_results[
        stats_results['Attack'].isin(attacks) & 
        stats_results['Aggregator'].isin(['FGC', 'FGC$_t$'])
    ].copy()

    # Add helper column for line styles
    plot_data['LineStyle'] = plot_data['Aggregator'].map(agg_map)

    # Plot with relplot
    g = sns.relplot(
        data=plot_data,
        x='f', 
        y='mean',
        hue='Attack',           # Color by attack
        hue_order=attacks,      # Fixed draw order for attacks
        style='Aggregator',  # Line style for TQ/TQ_t
        kind='line',
        markers=True,
        dashes={'FGC': (None, None), 'FGC$_t$': (2,2)},  # Solid/dashed
        height=4, aspect=2
    )
    # Show all spines
    sns.despine(left=False, bottom=False, right=False, top=False)

    g.set_axis_labels(r"Number of Byzantine workers $f$", "Accuracy (%)", fontsize=16)
    # g._legend.set_title("Attack and Aggregator")
    # Move legend to the right
    g._legend.set_bbox_to_anchor((0.99, 0.5))
    g._legend.set_loc("center left")

    # Set x and y axes
    g.set(xticks=[2, 4, 6], ylim=(80, 100))


    # Set y-axis tick interval to 5
    # for ax in g.axes.flat:
        # ax.set_yticks(range(80, 101, 5))  # From 80 to 100 with step 5
    
    # Set tick label font size
    for ax in g.axes.flat:
        ax.tick_params(axis='both', which='major', labelsize=16)
    
    # Set legend font size
    g._legend.get_title().set_fontsize(16)
    for text in g._legend.get_texts():
        text.set_fontsize(16)


    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR + "all_attacks_comparison.pdf", bbox_inches="tight", dpi=720)
    plt.show()


