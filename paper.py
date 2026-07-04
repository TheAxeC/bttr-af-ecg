"""Single entry point: rebuild every result and figure for the paper.

Run from this directory:
    python paper.py

Assumes PTB-XL 1.0.3 is fetched to ~/data/ptb-xl/ (set PTBXL_DIR in core/config.py; see README "Data").
Outputs land in results/*.csv (+ the block-map and forest .png/.npz); the two figures the paper uses
(results/forest_auroc.png and results/interpret_afsr_afibonly_blocks.png) are copied by hand into
../manuscript/figures/ before rebuilding the PDF.

The HuBERT-ECG step downloads the public foundation-model weights once (needs `transformers` + `torch` +
network); the CNN baseline needs `torch`. Both are optional dependencies of the benchmark, not of BTTR.
"""
import subprocess
import sys

STEPS = [
    ("HuBERT-ECG embeddings + linear probe (writes results/hubert_emb_*.npy)", ["-m", "experiments.hubert_probe"]),
    ("Table 1 - unified benchmark (BTTR vs CP vs CNN vs HuBERT vs clinical)", ["-m", "experiments.final_eval"]),
    ("forest plot", ["-m", "experiments.plot_forest"]),
    ("Table 2 (paired differences) + Table 3 CNN-alignment rows", ["-m", "experiments.paired_test"]),
    ("Table 3 BTTR rows (aligned/unaligned) + the K-sweep", ["-m", "experiments.harden_afsr"]),
    ("map-stability r + symmetric-tuning AUROC", ["-m", "experiments.run_analyses"]),
    ("block-map figures (primary + secondary)", ["-m", "experiments.interpret_afsr"]),
    ("temporal weight fractions (atrial/QRS/ST/T)", ["-m", "experiments.weight_fractions"]),
    ("detection-context cohort", ["-m", "experiments.run", "--task", "detect", "--per-class", "150"]),
    ("nested-CV robustness cross-check", ["-m", "experiments.nested_afsr"]),
]


def main():
    for label, args in STEPS:
        print(f"\n== {label} ==", flush=True)
        subprocess.run([sys.executable, *args], check=True)
    print("\nDONE -> results/.  Copy results/forest_auroc.png and "
          "results/interpret_afsr_afibonly_blocks.png into ../manuscript/figures/ before rebuilding the PDF.")


if __name__ == "__main__":
    main()
