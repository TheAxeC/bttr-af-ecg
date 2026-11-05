"""Write the paper's two figures as separate journal files.

Fig1 = forest_auroc   (AUROC with 95% patient-bootstrap CI, both cohorts; from final_table.csv)
Fig2 = interpret_afsr_afibonly_blocks   (BTTR block weight maps, primary cohort; from the committed .npz)

Each figure is written as Fig<N>.eps (vector, fonts embedded as Type 42) and Fig<N>.tif (RGB, 8 bit,
600 dpi, LZW) at a final width of 174 mm. The same drawing is also written as a 300 dpi RGB PNG into the
manuscript figures folder under the manuscript's file names (forest_auroc.png, interpret_afsr_afibonly_blocks.png),
so the in-text figures and the journal files show one layout. The drawing code is plot_forest.build_figure and
interpret.render_block_maps; only the page size, the wrapping of the Fig2 panels into two rows,
the lettering size (8 pt) and the removal of the in-figure suptitle differ from the manuscript PNGs.

Journal files go to $BTTR_FIGURES_DIR if set, else <project>/admin/submission/figures/ (the parent of code/).
The PNGs go to $BTTR_MANUSCRIPT_FIGURES_DIR if set, else <project>/manuscript/figures/; when that folder does
not exist (a code-only deposit) the PNGs are skipped and the run says so.

Run from code/:  python -m experiments.export_journal_figures
"""
import io
import os
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from PIL import Image

from core import config
from experiments import plot_forest
from experiments import interpret

FIGURES_DIR = Path(os.environ.get("BTTR_FIGURES_DIR") or
                   config.RESULTS_DIR.parent.parent / "admin" / "submission" / "figures")
MANUSCRIPT_FIGURES_DIR = Path(os.environ.get("BTTR_MANUSCRIPT_FIGURES_DIR") or
                              config.RESULTS_DIR.parent.parent / "manuscript" / "figures")
TIFF_DPI = 600
PNG_DPI = 300
MM = 25.4
WIDTH_MM = 174.0      # journal widths: 84, 129 or 174 mm
LETTER_PT = 8         # journal lettering: 8 to 12 pt at final size


def choose_font():
    """Return 'Arial' or 'Helvetica', whichever matplotlib can load; raise if neither."""
    for name in ("Arial", "Helvetica"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except ValueError:
            continue
    raise RuntimeError("Neither Arial nor Helvetica is installed; install one before exporting journal figures")


def rc(font_name, size=LETTER_PT):
    """rcParams for journal output: one sans-serif face, fonts embedded as TrueType (Type 42)."""
    return {"font.family": "sans-serif", "font.sans-serif": [font_name], "font.size": size,
            "axes.labelsize": size, "xtick.labelsize": size, "ytick.labelsize": size,
            "legend.fontsize": size, "axes.titlesize": size,
            "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False,
            "savefig.facecolor": "white"}


def rasterise(fig, dpi):
    """Render the figure at `dpi` to an 8 bit RGB PIL image on a white background."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def save_tiff(fig, path, dpi=TIFF_DPI):
    """Write an 8 bit RGB LZW TIFF that carries the dpi; return its pixel size."""
    img = rasterise(fig, dpi)
    img.save(path, format="TIFF", compression="tiff_lzw", dpi=(dpi, dpi))
    return img.size


def save_manuscript_png(fig, name):
    """Write <name>.png (RGB, PNG_DPI) into the manuscript figures folder, if that folder exists."""
    if not MANUSCRIPT_FIGURES_DIR.is_dir():
        print(f"skipped {name}.png: {MANUSCRIPT_FIGURES_DIR} does not exist")
        return
    img = rasterise(fig, PNG_DPI)
    img.save(MANUSCRIPT_FIGURES_DIR / f"{name}.png", format="PNG", dpi=(PNG_DPI, PNG_DPI))
    print(f"wrote {MANUSCRIPT_FIGURES_DIR.name}/{name}.png ({img.size[0]} x {img.size[1]} px at {PNG_DPI} dpi)")


def write_figure(fig, stem, png_name):
    """Write <stem>.eps and <stem>.tif into FIGURES_DIR, and <png_name>.png into the manuscript figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{stem}.eps", format="eps", facecolor="white")
    w, h = save_tiff(fig, FIGURES_DIR / f"{stem}.tif")
    print(f"wrote {stem}.eps and {stem}.tif ({w} x {h} px = {w / TIFF_DPI * MM:.1f} x {h / TIFF_DPI * MM:.1f} mm at {TIFF_DPI} dpi)")
    save_manuscript_png(fig, png_name)


def fig1(font_name):
    df = pd.read_csv(config.RESULTS_DIR / "final_table.csv")
    with plt.rc_context(rc(font_name)):
        fig = plot_forest.build_figure(df, figsize=(WIDTH_MM / MM, 2.9), ytick_fs=LETTER_PT, title_fs=LETTER_PT)
        write_figure(fig, "Fig1", "forest_auroc")
    plt.close(fig)


def fig2(font_name):
    z = np.load(config.RESULTS_DIR / "interpret_afsr_afibonly_blocks.npz")
    with plt.rc_context(rc(font_name)):
        fig = interpret.render_block_maps(
            z["maps"], z["b"], z["t"], list(z["leads"]), ncols=4, figsize=(WIDTH_MM / MM, 5.0),
            suptitle=False, tick_fs=LETTER_PT, title_fs=LETTER_PT, xlabel_fs=LETTER_PT, lead_labels="first_col")
        write_figure(fig, "Fig2", "interpret_afsr_afibonly_blocks")
    plt.close(fig)


def main():
    font_name = choose_font()
    print(f"font: {font_name}")
    fig1(font_name)
    fig2(font_name)


if __name__ == "__main__":
    main()
