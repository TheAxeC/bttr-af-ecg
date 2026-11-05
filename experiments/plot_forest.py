"""Forest plot of AUROC (95% patient-bootstrap CI) for all methods, both cohorts,
on the seed-0 draw, from final_table.csv."""
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from core import config


def build_figure(df, figsize=(9.6, 3.4), ytick_fs=8.5, title_fs=10):
    """Draw the forest plot from the final_table.csv frame and return the figure.

    The defaults reproduce forest_auroc.png; export_journal_figures.py passes the journal
    width and lettering sizes. Marks, colours, limits and strings do not depend on the arguments."""
    label = {"bttr": "BTTR", "hubert": "HuBERT-ECG probe",
             "cnn": "CNN", "clinical": "clinical features", "cp": "CP surrogate"}
    order = ["bttr", "hubert", "cnn", "clinical", "cp"]
    cohorts = [("AFIB (primary)", "AFIB (42 positives, 126 controls)"),
               ("AFIB+AFLT (secondary)", "AFIB or flutter (55 positives, 165 controls)")]
    colors = {"bttr": "#1b7837", "hubert": "#762a83", "cnn": "#9970ab", "clinical": "#d6604d", "cp": "#5aae61"}

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
    for ax, (key, title) in zip(axes, cohorts):
        d = df[df.cohort == key].set_index("method")
        ys = np.arange(len(order))[::-1]
        for y, m in zip(ys, order):
            mu, lo, hi = d.loc[m, "auroc"], d.loc[m, "ci_lo"], d.loc[m, "ci_hi"]
            lw = 2.6 if m == "bttr" else 2.0
            ax.plot([lo, hi], [y, y], color=colors[m], lw=lw, solid_capstyle="round")
            ax.plot(mu, y, "o", color=colors[m], ms=7 if m == "bttr" else 6)
        ax.axvline(0.5, ls=":", color="0.6", lw=1)
        ax.set_yticks(ys); ax.set_yticklabels([label[m] for m in order], fontsize=ytick_fs)
        ax.set_xlim(0.40, 0.85); ax.set_title(title, fontsize=title_fs)
        ax.set_xlabel("AUROC (95% patient-bootstrap CI)")
    axes[1].tick_params(labelleft=False)
    fig.tight_layout()
    return fig


def main():
    df = pd.read_csv(config.RESULTS_DIR / "final_table.csv")
    fig = build_figure(df)
    fig.savefig(config.RESULTS_DIR / "forest_auroc.png", dpi=200)
    print("wrote forest_auroc.png from final_table.csv")


if __name__ == "__main__":
    main()
