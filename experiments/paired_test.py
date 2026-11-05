"""Paired BTTR-minus-baseline AUROC differences (Table 2).

For each cohort, every method is scored out-of-fold on the seed-0 draw under the
same patient-grouped 5-fold split. For each baseline the script reports the point
estimate of the difference (BTTR AUROC minus baseline AUROC on the pooled
out-of-fold scores), the 95% percentile interval of the paired patient-level
bootstrap (patients resampled with replacement, both methods scored on the same
resample), and the fraction of resamples in which BTTR is higher.

The HuBERT-ECG probe is reported twice: with its fixed C=0.1, and with C chosen
by an inner patient-grouped 3-fold CV inside each training fold, so that no
setting of any baseline is chosen on test data.

Writes results/paired_test.csv.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from core import baselines as bl, config, preprocess as pp
from core.bttr_model import BTTRClassifier
from experiments.attia_cnn import cnn_oof
from experiments.hubert_probe import embeddings_for
from experiments.harden_afsr import _afsr_frames, _build_recs, load_db
K, SEED = 8, 0
C_GRID = (0.001, 0.01, 0.1, 1.0)


def _hubert_tuned_c(emb, y, g, tr):
    """Inner patient-grouped 3-fold CV on the training rows only; returns the best C."""
    best_c, best = C_GRID[0], -1.0
    inner = StratifiedGroupKFold(3, shuffle=True, random_state=SEED)
    for c in C_GRID:
        iy, isx = [], []
        for itr, ite in inner.split(np.zeros(len(tr)), y[tr], g[tr]):
            a, b = tr[itr], tr[ite]
            sc = StandardScaler().fit(emb[a])
            lr = LogisticRegression(max_iter=2000, C=c, random_state=SEED).fit(sc.transform(emb[a]), y[a])
            isx.append(lr.predict_proba(sc.transform(emb[b]))[:, 1]); iy.append(y[b])
        auc = roc_auc_score(np.concatenate(iy), np.concatenate(isx))
        if auc > best:
            best, best_c = auc, c
    return best_c

def oof(method, recs, T, emb, y, g):
    s=np.full(len(y),np.nan); sk=StratifiedGroupKFold(5,shuffle=True,random_state=SEED)
    for tr,te in sk.split(np.zeros(len(y)),y,g):
        if method=="bttr": s[te]=BTTRClassifier(nfactor=K).fit(T[tr],y[tr]).decision_function(T[te])
        elif method=="cp":
            Xtr,Xte=bl.cp_train_test_features(T[tr],T[te],rank=8,seed=SEED)
            sc=StandardScaler().fit(np.nan_to_num(Xtr)); lr=LogisticRegression(max_iter=1000,random_state=SEED).fit(sc.transform(np.nan_to_num(Xtr)),y[tr])
            s[te]=lr.predict_proba(sc.transform(np.nan_to_num(Xte)))[:,1]
        elif method=="clinical":
            Xtr=np.nan_to_num(bl.clinical_features([recs[i] for i in tr])); Xte=np.nan_to_num(bl.clinical_features([recs[i] for i in te]))
            sc=StandardScaler().fit(Xtr); lr=LogisticRegression(max_iter=1000,random_state=SEED).fit(sc.transform(Xtr),y[tr]); s[te]=lr.predict_proba(sc.transform(Xte))[:,1]
        elif method in ("hubert","hubert_tuned"):
            c=0.1 if method=="hubert" else _hubert_tuned_c(emb,y,g,tr)
            sc=StandardScaler().fit(emb[tr]); lr=LogisticRegression(max_iter=2000,C=c,random_state=SEED).fit(sc.transform(emb[tr]),y[tr]); s[te]=lr.predict_proba(sc.transform(emb[te]))[:,1]
    return s

def paired(scores,y,g,ref="bttr",n=2000):
    rng=np.random.default_rng(SEED); pats=np.unique(g); idx={p:np.where(g==p)[0] for p in pats}
    diffs={m:[] for m in scores if m!=ref}
    for _ in range(n):
        ch=rng.choice(pats,len(pats),True); ii=np.concatenate([idx[p] for p in ch])
        if len(np.unique(y[ii]))<2: continue
        a={m:roc_auc_score(y[ii],s[ii]) for m,s in scores.items()}
        for m in diffs: diffs[m].append(a[ref]-a[m])
    return {m:(float(np.mean(d)),float(np.percentile(d,2.5)),float(np.percentile(d,97.5)),float(np.mean(np.array(d)>0))) for m,d in diffs.items()}

def main():
    db=load_db(); rows=[]
    for inc,tag in ((False,"AFIB (primary)"),(True,"AFIB+AFLT (secondary)")):
        pos,healthy=_afsr_frames(db,inc); recs=_build_recs(pos,healthy,SEED)
        y=np.array([r.label for r in recs]); g=np.array([r.patient_id for r in recs])
        config.TENSORISATION="median_beat"; T=pp.build_tensor(recs)
        emb=embeddings_for(recs)
        scores={m:oof(m,recs,T,emb,y,g) for m in ("bttr","hubert","hubert_tuned","clinical","cp")}
        scores["cnn"]=cnn_oof(np.transpose(T,(0,2,1)).astype(np.float32),y,g)
        print(f"\n=== {tag}: paired BTTR - baseline AUROC difference (patient bootstrap) ===")
        pd_=paired(scores,y,g,"bttr")
        for m,(_,lo,hi,pg) in pd_.items():
            pt=roc_auc_score(y,scores["bttr"])-roc_auc_score(y,scores[m])
            print(f"  BTTR - {m:12s} delta={pt:+.3f} [{lo:+.3f},{hi:+.3f}]  P(BTTR>{m})={pg:.2f}")
            rows.append({"cohort":tag,"baseline":m,"delta":round(pt,3),"ci_lo":round(lo,3),"ci_hi":round(hi,3),"frac_bttr_higher":round(pg,2)})
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR/"paired_test.csv",index=False)
    print(f"\nWrote {config.RESULTS_DIR/'paired_test.csv'}")


if __name__=="__main__": main()
