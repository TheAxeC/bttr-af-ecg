"""Validating experiments (analysis only; no manuscript edits):
 (A) Paired patient-level bootstrap of AUROC differences BTTR - baseline (supports
     'on par / no accuracy cost': difference CI spanning 0 = not worse).
 (C) Symmetric CNN alignment: CNN on aligned (median_beat) vs unaligned (window)
     with patient-bootstrap CIs, matching BTTR's treatment.
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
from experiments.harden_afsr import _afsr_frames, _build_recs, _db_with_ondisk
K, SEED = 8, 0

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
        elif method=="hubert":
            sc=StandardScaler().fit(emb[tr]); lr=LogisticRegression(max_iter=2000,C=0.1,random_state=SEED).fit(sc.transform(emb[tr]),y[tr]); s[te]=lr.predict_proba(sc.transform(emb[te]))[:,1]
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

def boot1(s,y,g,n=2000):
    rng=np.random.default_rng(SEED); pats=np.unique(g); idx={p:np.where(g==p)[0] for p in pats}; a=[]
    for _ in range(n):
        ch=rng.choice(pats,len(pats),True); ii=np.concatenate([idx[p] for p in ch])
        if len(np.unique(y[ii]))==2: a.append(roc_auc_score(y[ii],s[ii]))
    return np.mean(a),np.percentile(a,2.5),np.percentile(a,97.5)

def main():
    db=_db_with_ondisk()
    for inc,tag,embf in ((False,"AFIB (primary)","afib"),(True,"AFIB+AFLT (secondary)","aflt")):
        pos,healthy=_afsr_frames(db,inc); recs=_build_recs(pos,healthy,SEED)
        y=np.array([r.label for r in recs]); g=np.array([r.patient_id for r in recs])
        config.TENSORISATION="median_beat"; T=pp.build_tensor(recs)
        emb=np.load(config.RESULTS_DIR/f"hubert_emb_{embf}.npy")
        scores={m:oof(m,recs,T,emb,y,g) for m in ("bttr","hubert","cnn_aligned","clinical","cp") if m not in ("cnn_aligned",)}
        scores["cnn_aligned"]=cnn_oof(np.transpose(T,(0,2,1)).astype(np.float32),y,g)
        print(f"\n=== {tag}: (A) paired BTTR - baseline AUROC difference (patient bootstrap) ===")
        pd_=paired({k:scores[k] for k in ("bttr","hubert","cnn_aligned","clinical","cp")},y,g,"bttr")
        for m,(dm,lo,hi,pg) in pd_.items():
            verdict="BTTR not worse" if lo>-0.05 else "check"
            print(f"  BTTR - {m:12s} delta={dm:+.3f} [{lo:+.3f},{hi:+.3f}]  P(BTTR>{m})={pg:.2f}")
        # (C) symmetric CNN alignment
        config.TENSORISATION="window"; W=pp.build_tensor(recs)
        cnn_un=cnn_oof(np.transpose(W,(0,2,1)).astype(np.float32),y,g)
        ca=boot1(scores["cnn_aligned"],y,g); cu=boot1(cnn_un,y,g)
        print(f"  (C) CNN aligned   AUROC={ca[0]:.3f} [{ca[1]:.3f},{ca[2]:.3f}]")
        print(f"  (C) CNN unaligned AUROC={cu[0]:.3f} [{cu[1]:.3f},{cu[2]:.3f}]   (BTTR unaligned ~0.45 = chance)")
    print("\ndone")

if __name__=="__main__": main()
