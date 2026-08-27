"""
ROC + Calibration curves — 4-panel figure for Inkscape layout
Design: ELSI W1+W2 train | ELSI W3 temporal | PNS 2013 retro | PNS 2019 prosp
Output: Outputs/Python_figures/fig_roc_calibration.pdf

Panel layout:
  [A] ROC — Pap smear     [B] ROC — Mammography
  [C] Calibration — Pap   [D] Calibration — Mama
"""
import warnings; warnings.filterwarnings("ignore")
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

# ==============================================================================
# PATHS
# ==============================================================================
ROOT     = Path(__file__).parent.parent.parent
DATA     = ROOT / "Data"
OUT_FIG  = ROOT / "Outputs" / "Python_figures"
MODELS   = ROOT / "Outputs" / "Models"
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

# ==============================================================================
# CONSTANTS
# ==============================================================================
MISS = {9,99,999,88,8}
def clean(s): return pd.to_numeric(s, errors="coerce").replace(list(MISS), np.nan)

ETHNIC_ELSI = {1:"White",2:"Black",3:"Brown",4:"Other",5:"Other"}
MAR_ELSI    = {1:"Without partner",2:"With partner",
               3:"Without partner",4:"Without partner"}
INS_ELSI    = {0:"No",1:"Yes"}
ZONE_R      = {1:"Urban",2:"Rural"}
REG_ELSI    = {1:"North",2:"Northeast",3:"Southeast",4:"South",5:"Central-West"}
MAR_PNS13   = {1:"With partner",2:"Without partner",3:"Without partner",
               4:"Without partner",5:"Without partner"}
MAR_PNS19   = {1:"With partner",2:"Without partner",
               3:"Without partner",4:"Without partner"}
INS_PNS     = {1:"Yes",2:"No"}
EDUC_PNS    = {1:"Illiterate",2:"Incomplete Elem",3:"Complete Elem",
               4:"Incomplete Elem",5:"Complete Elem",
               6:"Higher Educ",7:"Higher Educ"}
UF_REG      = {11:"North",12:"North",13:"North",14:"North",15:"North",
               16:"North",17:"North",21:"Northeast",22:"Northeast",
               23:"Northeast",24:"Northeast",25:"Northeast",26:"Northeast",
               27:"Northeast",28:"Northeast",29:"Northeast",31:"Southeast",
               32:"Southeast",33:"Southeast",35:"Southeast",41:"South",
               42:"South",43:"South",50:"Central-West",51:"Central-West",
               52:"Central-West",53:"Central-West"}

def educ_elsi(v):
    if pd.isna(v): return np.nan
    v = int(v)
    if v==1: return "Illiterate"
    if 2<=v<=8: return "Incomplete Elem"
    if 9<=v<=13: return "Complete Elem"
    if 14<=v<=18: return "Higher Educ"
    return np.nan

CAT_FEAT = ["age_group","region","zone","marital","ethnicity",
            "education","insurance"]
NUM_FEAT = ["n_residents"]
ALL_FEAT = CAT_FEAT + NUM_FEAT

# ==============================================================================
# DATA PREP
# ==============================================================================
def prep_elsi(path, wave):
    df, _ = pyreadstat.read_dta(str(path), encoding="latin1")
    o = pd.DataFrame()
    o["age"]        = clean(df["idade"])
    o["sex"]        = pd.to_numeric(df["sexo"], errors="coerce")
    o["ethnicity"]  = clean(df["e9"]).map(ETHNIC_ELSI)
    o["marital"]    = clean(df["e7"]).map(MAR_ELSI)
    o["zone"]       = clean(df["zona"]).map(ZONE_R)
    o["region"]     = clean(df["regiao"]).map(REG_ELSI)
    o["education"]  = clean(df["e22"]).apply(educ_elsi)
    o["insurance"]  = clean(df["u1"]).map(INS_ELSI)
    res = "ar6" if wave==1 else "nmoradores"
    o["n_residents"]= clean(df[res]).clip(upper=10)
    m13 = clean(df["m13"])
    o["target_pap"] = m13.isin({2,3}).astype("Int64")
    o["target_pap"] = o["target_pap"].where(m13.isin({1,2,3}), np.nan)
    m15 = clean(df["m15"])
    if wave in [1,2]:
        o["target_mamo"] = m15.isin({1,5}).astype("Int64")
    else:
        o["target_mamo"] = m15.isin({4,5}).astype("Int64")
    o["target_mamo"] = o["target_mamo"].where(m15.isin({1,2,3,4,5}), np.nan)
    return o[(o["sex"]==0) & (o["age"]>=50)].copy()

def prep_pns(path, sep, year):
    df = pd.read_csv(path, sep=sep, encoding="latin1", low_memory=False)
    o  = pd.DataFrame()
    o["age"]        = clean(df["C008"])
    o["sex"]        = clean(df["C006"])
    o["ethnicity"]  = clean(df["C009"]).map({1:"White",2:"Black",
                                              3:"Other",4:"Brown",5:"Other"})
    o["zone"]       = clean(df["V0026"]).map(ZONE_R)
    o["region"]     = pd.to_numeric(df["V0001"],errors="coerce").map(UF_REG)
    o["education"]  = clean(df["VDD004A"]).map(EDUC_PNS)
    o["n_residents"]= clean(df["V0022"]).clip(upper=10)
    if year==2013:
        o["marital"]  = clean(df["C011"]).map(MAR_PNS13)
        o["insurance"]= clean(df["I001"]).map(INS_PNS)
        pap = clean(df["R001"])
        o["target_pap"] = pap.isin([4,5]).astype("Int64")
        o["target_pap"] = o["target_pap"].where(pap.isin([1,2,3,4,5]), np.nan)
        r002 = pd.to_numeric(df["R002"], errors="coerce")
        o["target_pap"] = o["target_pap"].where(r002!=12, np.nan)
        md=clean(df["R015"]); mw=clean(df["R017"])
        o["target_mamo"] = ((md==2)|mw.isin([3,4])).astype("Int64")
        o["target_mamo"] = o["target_mamo"].where(
            md.isin([1,2])|mw.isin([1,2,3,4]), np.nan)
    else:
        o["marital"]  = clean(df["C011"]).map(MAR_PNS19)
        o["insurance"]= clean(df["I00102"]).map(INS_PNS)
        pap = clean(df["R00101"])
        o["target_pap"] = pap.isin([4,5]).astype("Int64")
        o["target_pap"] = o["target_pap"].where(pap.isin([1,2,3,4,5]), np.nan)
        r002010 = pd.to_numeric(df["R002010"], errors="coerce")
        o["target_pap"] = o["target_pap"].where(r002010!=12, np.nan)
        md=clean(df["R015"]); mw=clean(df["R01701"])
        o["target_mamo"] = ((md==2)|mw.isin([3,4])).astype("Int64")
        o["target_mamo"] = o["target_mamo"].where(
            md.isin([1,2])|mw.isin([1,2,3,4]), np.nan)
    return o[(o["sex"]==2) & (o["age"]>=50)].copy()

def get_Xy(df, target, age_min, age_max):
    s = df[(df["age"]>=age_min) & (df["age"]<=age_max)].copy()
    s["age_group"] = pd.cut(s["age"], bins=[age_min-1,59,age_max],
                            labels=["50-59",f"60-{age_max}"]).astype(str)
    sub = s[ALL_FEAT+[target]].dropna()
    return sub[ALL_FEAT].reset_index(drop=True), sub[target].astype(int).values

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
def plot_roc(ax, datasets_Xy, pipe, title, panel_label):
    """Plot ROC curves for all datasets on one axis."""
    ax.plot([0,1],[0,1], color="grey", lw=LW_DIAG,
            linestyle="--", alpha=0.6, zorder=0)

    for ds_label, (X,y) in datasets_Xy.items():
        yp  = pipe.predict_proba(X)[:,1]
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
        yp = pipe.predict_proba(X)[:,1]
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
    "Development performance based on full training set predictions.",
    ha="center", fontsize=7.5, color="grey", style="italic")

out_path = OUT_FIG / "fig_roc_calibration.pdf"
fig.savefig(out_path, format="pdf", bbox_inches="tight",
            dpi=300, facecolor="white")
plt.close()
print(f"Saved: {out_path}")
