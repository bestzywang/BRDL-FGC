r"""Exp:
- The following codes are used for Fig.5 (b)
"""
from utilscopy import get_args
from utilscopy import main
from utilscopy import EXP_DIR


args = get_args()
assert args.noniid
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
    EPOCHS = 100

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
        for agg in [ "avg", "rfa", "tm", "cm", "krum", "cp", "acp", "nacp" ]:
            for seed in [1, 2, 3]:
                for bucketing in [0]:
                    for momentum in [0.9]:
                        for attack in ["NA"]:
                            for n in [25]:
                                for f in [6]:
                                    for niid in [0.3, 0.5, 0.7]:
                                        for lr in [0.5]:
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
                        "niid": niid,
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

    # Define color mapping
    palette = {
        "FGC": "red",        # FGC in red
        "GM": "blue",       # GM in blue
        "CC": "green",      # CC in green
        "TM": "orange",     # TM in orange
        "CM": "purple",     # CM in purple
        "Average": "black",  # Average in black
        "Krum": "cyan"      # Krum in cyan
    }

    g = sns.relplot(
        data=results,
        x="Iterations",
        y="Accuracy (%)",
        col="niid",
        hue="Aggregator",
        height=3,
        aspect=1.5,
        palette=palette,
        kind="line",
        # col_wrap=2,  # This creates a 2x2 layout
    )

    g.set(xlim=(0, MAX_BATCHES_PER_EPOCH * EPOCHS), ylim=(0, 80))
    g.set_axis_labels("Iterations", "Accuracy (%)")
    g.set_titles(row_template=r"$|B| = {row_name}$", col_template=r"$\rho = {col_name}$")
    g.fig.subplots_adjust(wspace=0.1)

    g.fig.savefig(OUT_DIR + "exp33.pdf", bbox_inches="tight", dpi=720)
