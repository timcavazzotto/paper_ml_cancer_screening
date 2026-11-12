"""
ROC + Calibration curves — 4-panel figure for Inkscape layout
Design: ELSI W1+W2 train | ELSI W3 temporal | PNS 2013 retro | PNS 2019 prosp
Output: Outputs_v3/Python_figures/fig_roc_calibration.pdf

Panel layout:
  [A] ROC — Pap smear     [B] ROC — Mammography
  [C] Calibration — Pap   [D] Calibration — Mama
"""
import warnings; warnings.filterwarnings("ignore")
import os
from common import OUT_NAME, EDU_SCHEME
import numpy as np
import pandas as pd
import pyreadstat, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_predict

# ==============================================================================
# PATHS
# ==============================================================================
ROOT     = Path(__file__).parent.parent.parent
DATA     = ROOT / "Data"
OUT_FIG  = ROOT / OUT_NAME / "Python_figures"
MODELS   = ROOT / OUT_NAME / "Models"
OUT_FIG.mkdir(parents=True, exist_ok=True)

ELSI_W1 = DATA / "ELSI Portugues (1a onda) stata13.dta"
ELSI_W2 = DATA / "ELSI Portugues (2a onda) stata13.dta"
ELSI_W3 = DATA / "ELSI Portugues (3a onda).dta"
PNS13   = DATA / "PNS_2013.csv"
PNS19   = DATA / "pns2019.csv"

# ==============================================================================
# VISUAL STYLE
# ==============================================================================
# Consistent palette per dataset
PALETTE = {
    "ELSI W1+W2\n(development)":         "#1565C0",  # deep blue
    "ELSI W3\n(temporal val.)":           "#2E7D32",  # deep green
    "PNS 2013\n(ext. retro val.)":        "#E65100",  # deep orange
    "PNS 2019\n(ext. prosp val.)":        "#6A1B9A",  # deep purple
}
LINESTYLES = {
    "ELSI W1+W2\n(development)":         "-",
    "ELSI W3\n(temporal val.)":           "--",
    "PNS 2013\n(ext. retro val.)":        "-.",
    "PNS 2019\n(ext. prosp val.)":        ":",
}

# Typography
FONT_TITLE  = 16
FONT_LABEL  = 16
FONT_TICK   = 16
FONT_LEGEND = 12
FONT_ANNOT  = 14
LW          = 2   # line width
LW_DIAG     = 1.3   # diagonal reference


plt.rcParams.update({
    "font.family":       "Arial",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linewidth":    0.5,
})

from common_legacy import *      # regras únicas (common.py)

# ==============================================================================
# LOAD DATA
# ==============================================================================
print("Loading datasets...")
w1    = prep_elsi(ELSI_W1, 1)
w2    = prep_elsi(ELSI_W2, 2)
train = pd.concat([w1,w2], ignore_index=True)
w3    = prep_elsi(ELSI_W3, 3)
pns13 = prep_pns(PNS13, ";", 2013)
pns19 = prep_pns(PNS19, ",", 2019)
print("Done")

# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================
PRED_OVERRIDE = {}

def plot_roc(ax, datasets_Xy, pipe, title, panel_label):
    """Plot ROC curves for all datasets on one axis."""
    ax.plot([0,1],[0,1], color="grey", lw=LW_DIAG,
            linestyle="--", alpha=0.6, zorder=0)

    for ds_label, (X,y) in datasets_Xy.items():
        yp  = PRED_OVERRIDE.get(ds_label, pipe.predict_proba(X)[:,1])
        auc = roc_auc_score(y, yp)
        fpr, tpr, _ = roc_curve(y, yp)
        color = PALETTE[ds_label]
        ls    = LINESTYLES[ds_label]
        ax.plot(fpr, tpr, color=color, lw=LW, linestyle=ls,
                label=f"{ds_label.replace(chr(10),' ')} (AUC={auc:.3f})")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate", fontsize=FONT_LABEL)
    ax.set_ylabel("True positive rate", fontsize=FONT_LABEL)
    ax.set_title(f"{panel_label}  {title}", fontsize=FONT_TITLE,
                 fontweight="bold", loc="left")
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEGEND, loc="lower right",
              framealpha=0.9, edgecolor="grey")

def calibration_curve_trimmed(y, yp, n_bins=10, min_n=20):
    """Calibration curve trimming bins with fewer than min_n observations."""
    from sklearn.calibration import calibration_curve as _cal
    bins = np.linspace(0, 1, n_bins+1)
    mean_pred_list, frac_pos_list = [], []
    for i in range(n_bins):
        mask = (yp >= bins[i]) & (yp < bins[i+1])
        if i == n_bins-1:
            mask = (yp >= bins[i]) & (yp <= bins[i+1])
        n_bin = mask.sum()
        if n_bin >= min_n:
            mean_pred_list.append(yp[mask].mean())
            frac_pos_list.append(y[mask].mean())
    return np.array(mean_pred_list), np.array(frac_pos_list)

def plot_calibration(ax, datasets_Xy, pipe, title, panel_label,
                     n_bins=10, min_n=20):
    """Plot calibration curves, trimming bins with n < min_n."""
    ax.plot([0,1],[0,1], color="grey", lw=LW_DIAG,
            linestyle="--", alpha=0.6, zorder=0, label="Perfect calibration")

    for ds_label, (X,y) in datasets_Xy.items():
        yp = PRED_OVERRIDE.get(ds_label, pipe.predict_proba(X)[:,1])
        mean_pred, frac_pos = calibration_curve_trimmed(
            y, yp, n_bins=n_bins, min_n=min_n)
        if len(mean_pred) == 0:
            continue
        color = PALETTE[ds_label]
        ls    = LINESTYLES[ds_label]
        ax.plot(mean_pred, frac_pos, color=color, lw=LW, linestyle=ls,
                marker="o", markersize=4,
                label=ds_label.replace(chr(10)," "))

    # Axis limits based on data range
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Mean predicted probability", fontsize=FONT_LABEL)
    ax.set_ylabel("Fraction of positives", fontsize=FONT_LABEL)
    ax.set_title(f"{panel_label}  {title}", fontsize=FONT_TITLE,
                 fontweight="bold", loc="left")
    ax.tick_params(labelsize=FONT_TICK)
    # Add note about trimming
    ax.text(0.02, 0.97,
            f"Bins with n < {min_n} observations excluded",
            transform=ax.transAxes, fontsize=7, color="grey",
            style="italic", va="top")
    ax.legend(fontsize=FONT_LEGEND, loc="upper left",
              framealpha=0.9, edgecolor="grey")

# ==============================================================================
# BUILD FIGURE
# ==============================================================================
fig = plt.figure(figsize=(14, 11))
gs  = GridSpec(2, 2, figure=fig,
               hspace=0.38, wspace=0.32,
               left=0.08, right=0.97,
               top=0.95, bottom=0.07)

axes = [fig.add_subplot(gs[0,0]),  # A: ROC Pap
        fig.add_subplot(gs[0,1]),  # B: ROC Mamo
        fig.add_subplot(gs[1,0]),  # C: Cal Pap
        fig.add_subplot(gs[1,1])]  # D: Cal Mamo

for outcome, target, age_elsi, age_pns, ax_roc, ax_cal, panel_roc, panel_cal in [
    ("Pap smear",   "target_pap",  64, 64,
     axes[0], axes[2], "A", "C"),
    ("Mammography", "target_mamo", 69, 69,
     axes[1], axes[3], "B", "D"),
]:
    pipe = joblib.load(MODELS / f"model_{outcome.lower().replace(' ','_')}_XGBoost.joblib")

    datasets_Xy = {}
    for ds_label, df_ds, age_lim in [
        ("ELSI W1+W2\n(development)",   train, age_elsi),
        ("ELSI W3\n(temporal val.)",    w3,    age_elsi),
        ("PNS 2013\n(ext. retro val.)", pns13, age_pns),
        ("PNS 2019\n(ext. prosp val.)", pns19, age_pns),
    ]:
        X, y = get_Xy(df_ds, target, 50, age_lim)
        datasets_Xy[ds_label] = (X, y)
        if ds_label.startswith('ELSI W1+W2'):
            PRED_OVERRIDE[ds_label] = cross_val_predict(pipe, X, y, cv=CV, method='predict_proba')[:, 1]
        print(f"  {ds_label.replace(chr(10),' ')} {outcome}: n={len(y):,}")

    plot_roc(ax_roc, datasets_Xy, pipe,
             f"ROC \u2014 {outcome}", panel_roc)
    plot_calibration(ax_cal, datasets_Xy, pipe,
                     f"Calibration \u2014 {outcome}", panel_cal)
    print()

# Global note
fig.text(0.5, 0.005,
    "XGBoost model trained on ELSI-Brazil W1+W2 (2015\u20132021). "
    "Dashed diagonal: chance level (ROC) / perfect calibration. "
    "Development curve: out-of-fold predictions (5-fold CV).",
    ha="center", fontsize=7.5, color="grey", style="italic")

out_path = OUT_FIG / "fig_roc_calibration.pdf"
fig.savefig(out_path, format="pdf", bbox_inches="tight",
            dpi=300, facecolor="white")
plt.close()
print(f"Saved: {out_path}")
