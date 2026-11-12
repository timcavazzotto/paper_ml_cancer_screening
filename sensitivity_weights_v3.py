"""
Sensitivity analysis — survey sampling weights (Supplementary Table).
Same architecture as pns_pipeline_v3.py (preprocessing, algorithms, grids, 5-fold CV, seed 42).

Scenarios (cf. MacNell et al., PLoS One 2023;18:e0280387)
  U  unweighted training, unweighted evaluation   (primary analysis)
  W  weighted training (sample_weight), weighted evaluation
  P  unweighted training, weighted evaluation      (post hoc)
Weights: ELSI `peso_calibrado` (normalised to mean 1 within each wave before pooling W1+W2);
         PNS `V00291` (selected resident). Hyperparameter search scored by (unweighted) AUC.
Youden threshold: estimated on ELSI W3 (weighted in W/P) and applied to all datasets.
Output: Outputs_v3/Python_tables/sensitivity_weights.csv  (+ _docx)
"""
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, pandas as pd
from pathlib import Path
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from common_legacy import *
from common import get_Xy as get_Xy_weighted
from pns_pipeline_v3 import make_preprocessor, ALGO, GRIDS, CV

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_TAB = ROOT / OUT_NAME / "Python_tables"
N_ITER = int(os.environ.get("N_ITER", 40)); N_BOOT = int(os.environ.get("N_BOOT", 1000))

def thr_youden(y, p, w):
    f, t, th = roc_curve(y, p, sample_weight=w); return float(th[np.argmax(t - f)])

def metrics(y, p, thr, w):
    yc = (p >= thr).astype(int); w = np.ones(len(y)) if w is None else w
    tp = w[(yc == 1) & (y == 1)].sum(); fn = w[(yc == 0) & (y == 1)].sum()
    tn = w[(yc == 0) & (y == 0)].sum(); fp = w[(yc == 1) & (y == 0)].sum()
    return dict(auc=roc_auc_score(y, p, sample_weight=w), brier=brier_score_loss(y, p, sample_weight=w),
                sens=tp / (tp + fn), spec=tn / (tn + fp), prev=np.average(y, weights=w))

def boot_auc(y, p, w, n=N_BOOT, seed=42):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2: continue
        out.append(roc_auc_score(y[i], p[i], sample_weight=None if w is None else w[i]))
    return np.percentile(out, [2.5, 97.5])

def prep(df, target, age_max):
    """X, y, w with complete predictors AND a valid weight (same women in all scenarios)."""
    X, y, w = get_Xy_weighted(df, target, age_max, use_weights=True)
    return X, y, w

w1, w2, w3 = prep_elsi(ELSI_W1, 1), prep_elsi(ELSI_W2, 2), prep_elsi(ELSI_W3, 3)
train = pool_elsi_train(w1, w2)
sets = {"ELSI W3": w3, "PNS 2013": prep_pns(PNS13, ";", 2013), "PNS 2019": prep_pns(PNS19, ",", 2019)}
rows = []
for outcome, tgt, amax in [("Pap smear", "target_pap", PAP_MAX_AGE), ("Mammography", "target_mamo", MAMO_MAX_AGE)]:
    Xtr, ytr, wtr = prep(train, tgt, amax)
    full_n = {k: int(((d["age"] >= AGE_MIN) & (d["age"] <= amax) & d[tgt].notna()).sum()) for k, d in sets.items()}
    ev = {k: prep(d, tgt, amax) for k, d in sets.items()}
    for k, (X, y, w) in ev.items():
        print(f"  {outcome} {k}: n={len(y):,} (women with valid outcome in age range: {full_n[k]:,}; "
              f"complete predictors+weight)")
    for name in ["LogReg", "RandomForest", "XGBoost"]:
        base = clone(ALGO[name])
        if name == "XGBoost":
            base.set_params(scale_pos_weight=(ytr == 0).sum() / max((ytr == 1).sum(), 1))
        fits = {}
        for train_w in (False, True):
            rs = RandomizedSearchCV(Pipeline([("pre", make_preprocessor()), ("clf", base)]), GRIDS[name],
                                    n_iter=N_ITER, scoring="roc_auc", cv=CV, refit=True, n_jobs=-1, random_state=42)
            rs.fit(Xtr, ytr, **({"clf__sample_weight": wtr} if train_w else {}))
            fits[train_w] = rs.best_estimator_
        for scen, tw, ew in [("U", False, False), ("W", True, True), ("P", False, True)]:
            m = fits[tw]
            pw3 = m.predict_proba(ev["ELSI W3"][0])[:, 1]
            thr = thr_youden(ev["ELSI W3"][1], pw3, ev["ELSI W3"][2] if ew else None)
            for k, (X, y, w) in ev.items():
                p = m.predict_proba(X)[:, 1]; ww = w if ew else None
                r = metrics(y, p, thr, ww); lo, hi = boot_auc(y, p, ww)
                rows.append(dict(Outcome=outcome, Scenario=scen, Model=name, Dataset=k, n=len(y),
                                 Prev=f"{r['prev']*100:.1f}", AUC=f"{r['auc']:.3f} ({lo:.3f}–{hi:.3f})",
                                 Brier=f"{r['brier']:.3f}", Sens=f"{r['sens']:.3f}", Spec=f"{r['spec']:.3f}",
                                 Threshold=f"{thr:.3f}"))
        print(f"  done: {outcome} / {name}")
df = pd.DataFrame(rows); df.to_csv(OUT_TAB / "sensitivity_weights.csv", index=False)
print(df[df.Model == "XGBoost"].to_string(index=False))
