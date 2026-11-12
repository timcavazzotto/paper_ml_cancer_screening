"""
Training performance via 5-fold cross-validation (out-of-fold predictions)
Uses saved XGBoost models — no retraining needed
TRIPOD+AI item 16: performance on development dataset

Output: Outputs_v3/Python_tables/table2_train_cv_metrics.csv + .docx
"""
import warnings; warnings.filterwarnings("ignore")
import os
from common import OUT_NAME, EDU_SCHEME
import numpy as np
import pandas as pd
import pyreadstat, joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_auc_score, brier_score_loss,
                             confusion_matrix, roc_curve)
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ==============================================================================
# PATHS
# ==============================================================================
ROOT    = Path(__file__).parent.parent.parent
DATA    = ROOT / "Data"
OUT_TAB = ROOT / OUT_NAME / "Python_tables"
MODELS  = ROOT / OUT_NAME / "Models"
N_BOOT  = 1000
SEED    = 42
CV      = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

ELSI_W1 = DATA / "ELSI Portugues (1a onda) stata13.dta"
ELSI_W2 = DATA / "ELSI Portugues (2a onda) stata13.dta"

from common_legacy import *      # regras únicas (common.py)

# ==============================================================================
# METRICS
# ==============================================================================
def youden_thr(y, yp):
    fpr, tpr, thr = roc_curve(y, yp)
    return float(thr[np.argmax(tpr - fpr)])

def point_metrics(y, yp, thr):
    yc = (yp >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yc).ravel()
    return dict(
        AUC   = round(roc_auc_score(y, yp), 3),
        Brier = round(brier_score_loss(y, yp), 3),
        Sens  = round(tp/(tp+fn), 3) if (tp+fn)>0 else np.nan,
        Spec  = round(tn/(tn+fp), 3) if (tn+fp)>0 else np.nan,
        PPV   = round(tp/(tp+fp), 3) if (tp+fp)>0 else np.nan,
        NPV   = round(tn/(tn+fn), 3) if (tn+fn)>0 else np.nan,
    )

def bootstrap_ci(y, yp, thr, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); yp = np.asarray(yp)
    acc = {k:[] for k in ["AUC","Brier","Sens","Spec","PPV","NPV"]}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        yb, ypb = y[idx], yp[idx]
        if len(np.unique(yb)) < 2: continue
        for k, v in point_metrics(yb, ypb, thr).items():
            if not np.isnan(v): acc[k].append(v)
    ci = {}
    for k, vals in acc.items():
        if vals:
            ci[f"{k}_lo"] = round(np.percentile(vals, 2.5), 3)
            ci[f"{k}_hi"] = round(np.percentile(vals, 97.5), 3)
        else:
            ci[f"{k}_lo"] = ci[f"{k}_hi"] = np.nan
    return ci

def fmt(p, lo, hi):
    return f"{p:.3f} ({lo:.3f}\u2013{hi:.3f})"

# ==============================================================================
# DOCX HELPERS
# ==============================================================================
def set_bg(cell, color):
    tc=cell._tc; pr=tc.get_or_add_tcPr()
    shd=OxmlElement("w:shd")
    shd.set(qn("w:val"),"clear"); shd.set(qn("w:color"),"auto")
    shd.set(qn("w:fill"),color); pr.append(shd)

def set_txt(cell, text, bold=False, size=8,
            align=WD_ALIGN_PARAGRAPH.CENTER):
    cell.text=""
    p=cell.paragraphs[0]; p.alignment=align
    r=p.add_run(str(text))
    r.bold=bold; r.font.size=Pt(size); r.font.name="Times New Roman"

# ==============================================================================
# MAIN
# ==============================================================================
print("Loading ELSI W1+W2 training pool...")
w1 = prep_elsi(ELSI_W1, 1)
w2 = prep_elsi(ELSI_W2, 2)
train = pd.concat([w1, w2], ignore_index=True)
print(f"  Pool: {len(train):,} women")

rows = []

for outcome, target, age_elsi in [
    ("Pap smear",   "target_pap",  64),
    ("Mammography", "target_mamo", 69),
]:
    print(f"\n{outcome}:")
    pipe = joblib.load(MODELS / f"model_{outcome.lower().replace(' ','_')}_XGBoost.joblib")

    X_tr, y_tr = get_Xy(train, target, 50, age_elsi)
    print(f"  n={len(y_tr):,} | prevalence={y_tr.mean():.1%}")

    # Out-of-fold CV predictions
    print(f"  Running 5-fold cross_val_predict...", end=" ", flush=True)
    yp_oof = cross_val_predict(pipe, X_tr, y_tr,
                               cv=CV, method="predict_proba")[:, 1]
    print("done")

    # Youden threshold from OOF predictions
    thr = youden_thr(y_tr, yp_oof)
    print(f"  Youden threshold (OOF): {thr:.3f}")

    # Point metrics
    m = point_metrics(y_tr, yp_oof, thr)

    # Bootstrap CI
    print(f"  Bootstrap CI (n={N_BOOT})...", end=" ", flush=True)
    ci = bootstrap_ci(y_tr, yp_oof, thr)
    print("done")

    # Also compute per-fold AUC for SD
    fold_aucs = []
    # Out-of-fold predictions of each validation fold (the CV object is deterministic, so the
    # folds are identical to those of cross_val_predict). The previous version used the final
    # model fitted on ALL data, which made the per-fold AUC an in-sample (optimistic) estimate.
    for train_idx, val_idx in CV.split(X_tr, y_tr):
        y_f, yp_f = y_tr[val_idx], yp_oof[val_idx]
        if len(np.unique(y_f)) >= 2:
            fold_aucs.append(roc_auc_score(y_f, yp_f))
    auc_cv_mean = np.mean(fold_aucs)
    auc_cv_std  = np.std(fold_aucs)
    print(f"  AUC CV (fold mean±SD): {auc_cv_mean:.3f}±{auc_cv_std:.3f}")
    print(f"  AUC OOF: {m['AUC']:.3f} ({ci['AUC_lo']:.3f}–{ci['AUC_hi']:.3f})")

    rows.append({
        "Outcome":  outcome,
        "Dataset":  f"ELSI W1+W2\n(development, 5-fold CV)",
        "n":        len(y_tr),
        "Prev (%)": f"{y_tr.mean()*100:.1f}",
        "AUC CV (fold mean\u00b1SD)": f"{auc_cv_mean:.3f}\u00b1{auc_cv_std:.3f}",
        "AUC OOF (95% CI)":   fmt(m["AUC"],   ci["AUC_lo"],   ci["AUC_hi"]),
        "Brier (95% CI)":     fmt(m["Brier"], ci["Brier_lo"], ci["Brier_hi"]),
        "Sens (95% CI)":      fmt(m["Sens"],  ci["Sens_lo"],  ci["Sens_hi"]),
        "Spec (95% CI)":      fmt(m["Spec"],  ci["Spec_lo"],  ci["Spec_hi"]),
        "PPV (95% CI)":       fmt(m["PPV"],   ci["PPV_lo"],   ci["PPV_hi"]),
        "NPV (95% CI)":       fmt(m["NPV"],   ci["NPV_lo"],   ci["NPV_hi"]),
        "Threshold":          f"{thr:.3f} (Youden, OOF)",
    })

# Save CSV
df_out = pd.DataFrame(rows)
df_out.to_csv(OUT_TAB / "table2_train_cv_metrics.csv", index=False)
print(f"\nCSV saved: table2_train_cv_metrics.csv")
print(df_out[["Outcome","n","AUC CV (fold mean\u00b1SD)",
              "AUC OOF (95% CI)","Sens (95% CI)","Spec (95% CI)"]].to_string(index=False))

# ==============================================================================
# WORD TABLE
# ==============================================================================
doc = Document()
for sec in doc.sections:
    sec.top_margin=Cm(2); sec.bottom_margin=Cm(2)
    sec.left_margin=Cm(2); sec.right_margin=Cm(2)

tp = doc.add_paragraph()
tr = tp.add_run(
    "Table 2 (supplement) — Predictive performance of XGBoost model "
    "on the development dataset (ELSI W1+W2, 5-fold cross-validation).")
tr.bold=True; tr.font.size=Pt(10); tr.font.name="Times New Roman"
doc.add_paragraph()

COLS = [
    ("Outcome",                    "Outcome",                    Cm(2.4), WD_ALIGN_PARAGRAPH.LEFT),
    ("n",                          "n",                          Cm(1.2), WD_ALIGN_PARAGRAPH.CENTER),
    ("Prev\n(%)",                  "Prev (%)",                   Cm(0.9), WD_ALIGN_PARAGRAPH.CENTER),
    ("AUC CV\n(fold mean\u00b1SD)","AUC CV (fold mean\u00b1SD)", Cm(2.8), WD_ALIGN_PARAGRAPH.CENTER),
    ("AUC OOF\n(95% CI)",          "AUC OOF (95% CI)",           Cm(3.2), WD_ALIGN_PARAGRAPH.CENTER),
    ("Brier\n(95% CI)",            "Brier (95% CI)",             Cm(3.0), WD_ALIGN_PARAGRAPH.CENTER),
    ("Sens\n(95% CI)",             "Sens (95% CI)",              Cm(3.0), WD_ALIGN_PARAGRAPH.CENTER),
    ("Spec\n(95% CI)",             "Spec (95% CI)",              Cm(3.0), WD_ALIGN_PARAGRAPH.CENTER),
    ("PPV\n(95% CI)",              "PPV (95% CI)",               Cm(3.0), WD_ALIGN_PARAGRAPH.CENTER),
    ("NPV\n(95% CI)",              "NPV (95% CI)",               Cm(3.0), WD_ALIGN_PARAGRAPH.CENTER),
    ("Threshold",                  "Threshold",                  Cm(2.4), WD_ALIGN_PARAGRAPH.CENTER),
]

t = doc.add_table(rows=1+len(rows), cols=len(COLS))
t.style = "Table Grid"
for ci, (_,_,w,_) in enumerate(COLS):
    for cell in t.columns[ci].cells: cell.width = w

hdr = t.rows[0].cells
for ci, (h,_,_,a) in enumerate(COLS):
    set_txt(hdr[ci], h, bold=True, size=8); set_bg(hdr[ci], "D9D9D9")

prev_out = None
for ri, rd in enumerate(rows, start=1):
    row = t.rows[ri].cells
    bg  = "F5F5F5" if rd["Outcome"]=="Pap smear" else "FFFFFF"
    for ci, (_,key,_,a) in enumerate(COLS):
        val = rd.get(key, "")
        if key=="Outcome":
            val = val if val!=prev_out else ""; prev_out=rd["Outcome"]
        set_txt(row[ci], str(val), size=8, align=a)
        set_bg(row[ci], bg)

doc.add_paragraph()
fn = doc.add_paragraph()
fn_r = fn.add_run(
    "AUC CV: mean AUC across 5 validation folds (reported as mean\u00b1SD). "
    "AUC OOF: AUC computed on pooled out-of-fold predictions. "
    "Brier: Brier score. Sens: sensitivity. Spec: specificity. "
    "PPV/NPV: predictive values. "
    "95% CI: non-parametric bootstrap (n=1,000 resamples) applied to OOF predictions. "
    "Threshold: Youden index estimated on OOF predictions. "
    "XGBoost model trained on ELSI W1+W2 pool (n=4,779 Pap; n=6,335 mammography).")
fn_r.font.size=Pt(8); fn_r.font.name="Times New Roman"
fn.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY

doc.save(OUT_TAB / "table2_train_cv_metrics.docx")
print("Word saved: table2_train_cv_metrics.docx")
