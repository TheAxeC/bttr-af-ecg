"""Probe the HuBERT-ECG foundation model (pretrained on 9.1M ECGs) on AF-in-SR:
is it a strong TEACHER (stronger than BTTR)? Linear probe on mean-pooled embeddings,
patient-grouped 5-fold, patient-bootstrap CIs. If strong -> distill into a glass-box
tensor student (the novel method).
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from core import config
from core import preprocess as pp
from core.bttr_model import BTTRClassifier
from experiments.harden_afsr import _afsr_frames, _build_recs, load_db

SEED = 0
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEC = 5  # seconds at 100 Hz


def load_hubert():
    """Load pretrained weights into trusted transformers HubertModel (no remote code)."""
    import json
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import HubertConfig, HubertModel
    repo = "Edoardo-BS/hubert-ecg-small"
    cfg = HubertConfig.from_dict(json.load(open(hf_hub_download(repo, "config.json"))))
    model = HubertModel(cfg)
    sd = load_file(hf_hub_download(repo, "model.safetensors"))
    sd = {k: v for k, v in sd.items() if not (k.startswith("final_proj") or k.startswith("label_embedding"))}
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"loaded HuBERT-ECG: {len(sd)} tensors, missing={len(miss)} unexpected={len(unexp)}")
    return model.to(DEV).eval()


def embed(model, recs, fs=100):
    """Each record -> concat 12-lead 5s@100Hz, per-record z-norm -> mean-pooled 512-d."""
    embs = []
    with torch.no_grad():
        for r in recs:
            sig = pp.clean(r.signal, r.fs, decimate_factor=1)        # (T,12) ~100Hz
            n = SEC * fs
            x = sig[:n] if len(sig) >= n else np.pad(sig, ((0, n - len(sig)), (0, 0)))
            x = x.T.reshape(-1)                                       # concat leads -> (12*n,)
            x = (x - x.mean()) / (x.std() + 1e-6)
            iv = torch.tensor(x, dtype=torch.float32, device=DEV)[None]
            out = model(input_values=iv).last_hidden_state           # (1, frames, 512)
            embs.append(out.mean(1).squeeze(0).cpu().numpy())
    return np.array(embs)


EMB_CACHE = config.RESULTS_DIR / "hubert_emb.npz"
_MODEL = None


def embeddings_for(recs):
    """HuBERT-ECG embeddings of `recs`, in their order, from a cache keyed by ecg_id.

    Records not yet in results/hubert_emb.npz are embedded with the frozen model and
    added to the cache, so any control draw can be scored."""
    global _MODEL
    cache = {}
    if EMB_CACHE.exists():
        z = np.load(EMB_CACHE)
        cache = dict(zip(z["ecg_id"].tolist(), z["emb"]))
    todo = [r for r in recs if int(r.record_id) not in cache]
    if todo:
        if _MODEL is None:
            _MODEL = load_hubert()
        for r, e in zip(todo, embed(_MODEL, todo)):
            cache[int(r.record_id)] = e
        ids = sorted(cache)
        np.savez(EMB_CACHE, ecg_id=np.array(ids), emb=np.stack([cache[i] for i in ids]))
    return np.stack([cache[int(r.record_id)] for r in recs])


def oof(method, recs, X, emb, y, groups):
    s = np.full(len(y), np.nan)
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    for tr, te in outer.split(np.zeros(len(y)), y, groups):
        if method == "bttr":
            s[te] = BTTRClassifier(nfactor=8).fit(X[tr], y[tr]).decision_function(X[te])
        else:  # hubert linear probe
            sc = StandardScaler().fit(emb[tr])
            lr = LogisticRegression(max_iter=2000, C=0.1, random_state=SEED).fit(sc.transform(emb[tr]), y[tr])
            s[te] = lr.predict_proba(sc.transform(emb[te]))[:, 1]
    return s


def boot(scores, y, g, ref="bttr", n=2000):
    rng = np.random.default_rng(SEED); pats = np.unique(g)
    idx = {p: np.where(g == p)[0] for p in pats}; coll = {m: [] for m in scores}; dl = {m: [] for m in scores}
    for _ in range(n):
        ch = rng.choice(pats, len(pats), True); ii = np.concatenate([idx[p] for p in ch])
        if len(np.unique(y[ii])) < 2:
            continue
        a = {m: roc_auc_score(y[ii], s[ii]) for m, s in scores.items()}
        for m in scores:
            coll[m].append(a[m]); dl[m].append(a[m] - a[ref])
    pt = {m: float(roc_auc_score(y, s)) for m, s in scores.items()}
    return {m: (pt[m], float(np.percentile(coll[m], 2.5)), float(np.percentile(coll[m], 97.5)),
               pt[m] - pt[ref], float(np.mean(np.array(dl[m]) > 0))) for m in scores}


def main():
    print(f"device={DEV}")
    db = load_db(); rows = []
    for inc, tag in ((False, "AFIB (primary)"), (True, "AFIB+AFLT (secondary)")):
        pos, healthy = _afsr_frames(db, inc); recs = _build_recs(pos, healthy, SEED)
        y = np.array([r.label for r in recs]); g = np.array([r.patient_id for r in recs])
        config.TENSORISATION = "median_beat"; X = pp.build_tensor(recs)
        emb = embeddings_for(recs)
        print(f"[{tag}] emb shape {emb.shape}")
        scores = {"bttr": oof("bttr", recs, X, emb, y, g), "hubert": oof("hubert", recs, X, emb, y, g)}
        ci = boot(scores, y, g, "bttr")
        for m in ("bttr", "hubert"):
            mu, lo, hi, dm, dp = ci[m]; ap = average_precision_score(y, scores[m])
            flag = "  <-- STRONG TEACHER" if (m == "hubert" and dp > 0.8) else ""
            print(f"  {m:8s} AUROC={mu:.3f} [{lo:.3f},{hi:.3f}] AUPRC={ap:.3f}  dvs_bttr={dm:+.3f} P(>0)={dp:.2f}{flag}")
            rows.append({"cohort": tag, "method": m, "auroc": mu, "ci_lo": lo, "ci_hi": hi, "auprc": ap, "d_vs_bttr": dm, "p_gt0": dp})
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "hubert_probe.csv", index=False)
    print(f"\nWrote {config.RESULTS_DIR / 'hubert_probe.csv'}")


if __name__ == "__main__":
    main()
