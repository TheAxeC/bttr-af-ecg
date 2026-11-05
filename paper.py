"""Single entry point: rebuild every result and figure for the paper.

Run from this directory:
    python paper.py

Assumes PTB-XL 1.0.3 (ptbxl_database.csv and records100/) is downloaded to ~/data/ptb-xl/ (set PTBXL_DIR in
core/config.py; see README "Data"). Outputs land in results/ (CSVs, block-map .npz/.png, the HuBERT embedding
cache) and manifests/ (the record-level cohorts and folds); export_journal_figures writes the two manuscript
figures into ../manuscript/figures/ and the journal EPS/TIFF files into ../admin/submission/figures/.

The HuBERT-ECG step downloads the public foundation-model weights once (needs `transformers` + `torch` +
network); the CNN baseline needs `torch`. Both are optional dependencies of the benchmark, not of BTTR.
"""
import subprocess
import sys

STEPS = [
    ("record-level cohort and split manifests (manifests/)", ["-m", "experiments.export_manifests"]),
    ("HuBERT-ECG embeddings + linear probe (writes the embedding cache results/hubert_emb.npz)", ["-m", "experiments.hubert_probe"]),
    ("Table 1 - benchmark on the seed-0 draw (BTTR vs CP vs CNN vs HuBERT vs clinical)", ["-m", "experiments.final_eval"]),
    ("forest plot", ["-m", "experiments.plot_forest"]),
    ("Table 2 - paired differences with 95% intervals", ["-m", "experiments.paired_test"]),
    ("all methods over the ten matched control draws", ["-m", "experiments.benchmark_draws"]),
    ("Table 3 - alignment ablation, BTTR and CNN under both protocols", ["-m", "experiments.matched_ablation"]),
    ("paired CIs of the ten-draw mean differences, methods and alignment (pooled patient bootstrap)", ["-m", "experiments.pooled_paired"]),
    ("tensorisation comparison + the K sweep (ten draws)", ["-m", "experiments.harden_afsr"]),
    ("map-stability r + symmetric-tuning AUROC", ["-m", "experiments.run_analyses"]),
    ("block-map figures (primary + secondary)", ["-m", "experiments.interpret_afsr"]),
    ("temporal weight fractions (atrial/QRS/ST/T)", ["-m", "experiments.weight_fractions"]),
    ("detection-context cohort", ["-m", "experiments.run", "--task", "detect", "--per-class", "150"]),
    ("nested-CV robustness cross-check", ["-m", "experiments.nested_afsr"]),
    ("journal figures (manuscript PNGs + EPS/TIFF)", ["-m", "experiments.export_journal_figures"]),
    ("graphical abstract", ["-m", "experiments.graphical_abstract"]),
]


def main():
    for label, args in STEPS:
        print(f"\n== {label} ==", flush=True)
        subprocess.run([sys.executable, *args], check=True)
    print("\nDONE -> results/, manifests/ and the manuscript figures.")


if __name__ == "__main__":
    main()
