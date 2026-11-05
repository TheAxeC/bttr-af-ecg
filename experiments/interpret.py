"""Interpretability export - the paper's lead figure.

Fits BTTR on the full cohort and saves, for each block k:
  * a (time x lead) heatmap of the weight map (the spatio-temporal AF signature),
  * the per-lead profile and the temporal profile (marginals),
  * the regression coefficient b_k (how strongly the block drives the AF score).

Because BTTR is supervised, these maps are discriminative by construction: bright
regions are where the model reads AF-vs-SR evidence (e.g. fibrillatory activity in
the TP/PR segment, RR-driven structure). This is the glass-box the thesis promised
but never produced (it only ran unsupervised CP).
"""
from __future__ import annotations

import numpy as np

from core import config
from core import preprocess as pp
from core.bttr_model import BTTRClassifier


def render_block_maps(maps, b, t, names, ncols=None, figsize=None, suptitle=True,
                      tick_fs=7, title_fs=9, xlabel_fs=None, lead_labels="all"):
    """Draw one (time x lead) heatmap per block, most influential block (largest |b|) first.

    The defaults reproduce the <prefix>_blocks.png layout: a single row, a suptitle, lead labels on
    every panel. ncols wraps the panels into a grid; lead_labels="first_col" labels the leads only on
    the first column of the grid. Colours, colour limits, ordering and strings do not depend on the arguments."""
    import matplotlib.pyplot as plt

    K = maps.shape[0]
    order = np.argsort(-np.abs(b))                       # most influential blocks first
    ncols = ncols or K
    nrows = -(-K // ncols)
    figsize = figsize or (3.2 * K, 4)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    vmax = np.percentile(np.abs(maps), 99) or 1.0
    for i, k in enumerate(order):
        ax = axes[i // ncols][i % ncols]
        ax.imshow(maps[k].T, aspect="auto", cmap="PiYG", vmin=-vmax, vmax=vmax,
                  extent=[t[0], t[-1], len(names) - 0.5, -0.5])
        ax.set_yticks(range(len(names)))
        if lead_labels == "all" or i % ncols == 0:
            ax.set_yticklabels(names, fontsize=tick_fs)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("time (s)", **({} if xlabel_fs is None else {"fontsize": xlabel_fs}))
        ax.set_title(f"block {k}  b={b[k]:+.2f}", fontsize=title_fs)
    for j in range(K, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    if suptitle:
        fig.suptitle("BTTR block weight maps - spatio-temporal AF signatures")
    fig.tight_layout()
    return fig


def export(recordings, y, out_prefix="interpret"):
    X = pp.build_tensor(recordings)
    clf = BTTRClassifier().fit(X, y)
    maps, b = clf.block_maps()                          # (K, T, L), (K,)
    T = maps.shape[1]
    t = pp.time_axis_seconds(T)

    np.savez(config.RESULTS_DIR / f"{out_prefix}_blocks.npz",
             maps=maps, b=b, t=t, leads=np.array(config.LEAD_NAMES))

    # optional rendering (skipped headlessly if matplotlib is absent)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print(f"[interpret] saved factors to {out_prefix}_blocks.npz (no matplotlib)")
        return maps, b

    fig = render_block_maps(maps, b, t, pp.lead_names())    # includes RR pseudo-leads if used
    fig.savefig(config.RESULTS_DIR / f"{out_prefix}_blocks.png", dpi=150)
    print(f"[interpret] wrote {out_prefix}_blocks.png and .npz")
    return maps, b
