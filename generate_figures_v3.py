"""
Generate publication figures:
  1. SHAP beeswarm with clean variable labels
  2. Barrier analysis figure (stacked bars by risk tertile)
  3. Decision curve analysis (DCA)

Output: Outputs_v3/Python_figures/
  fig_shap_pap_smear.pdf
  fig_shap_mammography.pdf
  fig_barrier_analysis.pdf
  fig_dca.pdf
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import shap

# ==============================================================================
# PATHS
# ==============================================================================
ROOT    = Path(__file__).parent.parent.parent
DATA    = ROOT / "Data"
OUT_FIG = ROOT / OUT_NAME / "Python_figures"
OUT_TAB = ROOT / OUT_NAME / "Python_tables"
MODELS  = ROOT / OUT_NAME / "Models"
OUT_FIG.mkdir(parents=True, exist_ok=True)

ELSI_W1 = DATA / "ELSI Portugues (1a onda) stata13.dta"
ELSI_W2 = DATA / "ELSI Portugues (2a onda) stata13.dta"
ELSI_W3 = DATA / "ELSI Portugues (3a onda).dta"
PNS13   = DATA / "PNS_2013.csv"
PNS19   = DATA / "pns2019.csv"

# ==============================================================================
# VISUAL STYLE
# ==============================================================================
plt.rcParams.update({
    "font.family":       "Arial",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linewidth":    0.5,
})
FONT_TITLE  = 16
FONT_LABEL  = 14
FONT_TICK   = 12
FONT_LEGEND = 10

# ==============================================================================
# CLEAN SHAP LABEL MAPPING
# OHE format → readable label
# ==============================================================================
SHAP_LABELS = {
    # Age group
    "age_group_50-59":           "Age group: 50–59 years",
    "age_group_60-64":           "Age group: 60–64 years",
    "age_group_60-69":           "Age group: 60–69 years",
    # Region
    "region_North":              "Geographic region: North",
    "region_Northeast":          "Geographic region: Northeast",
    "region_Southeast":          "Geographic region: Southeast",
    "region_South":              "Geographic region: South",
    "region_Central-West":       "Geographic region: Central-West",
    # Zone
    "zone_Urban":                "Area of residence: Urban",
    "zone_Rural":                "Area of residence: Rural",
    # Marital
    "marital_With partner":      "Marital status: With partner",
    "marital_Without partner":   "Marital status: Without partner",
    # Ethnicity
    "ethnicity_White":           "Race/ethnicity: White",
    "ethnicity_Black":           "Race/ethnicity: Black",
    "ethnicity_Brown":           "Race/ethnicity: Brown",
    "ethnicity_Other":           "Race/ethnicity: Other\u1d43",
    # Education
    "education_Higher education":                "Education: Higher education",
    "education_Complete elementary to secondary": "Education: Complete elementary to secondary",
    "education_Up to incomplete elementary":     "Education: Up to incomplete elementary",
    "education_No schooling":                    "Education: No schooling",
    "education_Incomplete elementary":           "Education: Incomplete elementary",
    # Insurance
    "insurance_Yes":             "Health insurance: Yes",
    "insurance_No":              "Health insurance: No",
    # N residents
    "n_residents":               "Household size (n residents)",
}

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
print("Done\n")

# ==============================================================================
# 1. SHAP WITH CLEAN LABELS
# ==============================================================================
def plot_shap(outcome, target, age_elsi, panel_label):
    print(f"SHAP — {outcome}...")
    pipe = joblib.load(MODELS/f"model_{outcome.lower().replace(' ','_')}_XGBoost.joblib")
    X_w3, y_w3 = get_Xy(w3, target, 50, age_elsi)

    # Get feature names from preprocessor
    ohe  = pipe["pre"].named_transformers_["cat"]["ohe"]
    feat_names = ohe.get_feature_names_out(CAT_FEAT).tolist() + NUM_FEAT

    # Transform
    X_t = pd.DataFrame(pipe["pre"].transform(X_w3), columns=feat_names)

    # SHAP values
    expl = shap.TreeExplainer(pipe["clf"])
    sv   = expl.shap_values(X_t)
    if isinstance(sv, list): sv = sv[1]
    if sv.ndim == 3: sv = sv[:,:,1]

    # Rename features
    clean_names = [SHAP_LABELS.get(f, f) for f in feat_names]
    X_clean = X_t.copy()
    X_clean.columns = clean_names

    # Compute mean |SHAP| per feature and keep top 15
    mean_abs = np.abs(sv).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[-15:][::-1]
    sv_top   = sv[:, top_idx]
    X_top    = X_clean.iloc[:, top_idx]

    fig, ax = plt.subplots(figsize=(7, 7))
    shap.summary_plot(sv_top, X_top, max_display=15,
                      show=False, plot_size=None, color_bar=True)
    ax = plt.gca()
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=FONT_LABEL)
    ax.set_title(f"{panel_label}  {outcome} — Non-adherence\n"
                 f"XGBoost, ELSI W3 temporal validation set (n={len(y_w3):,})",
                 fontsize=FONT_TITLE, fontweight="bold", loc="left")
    ax.tick_params(labelsize=FONT_TICK)

    # Footnote for Other
    fig.text(0.12, 0.01,
             "\u1d43 Other includes Yellow and Indigenous categories "
             "(collapsed due to small cell sizes).",
             fontsize=7, color="grey", style="italic")

    out = OUT_FIG / f"fig_shap_{outcome.lower().replace(' ','_')}.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight",
                facecolor="white", dpi=300)
    plt.close()
    print(f"  Saved: {out.name}")

plot_shap("Pap smear",   "target_pap",  64, "A")
plot_shap("Mammography", "target_mamo", 69, "B")

# ==============================================================================
# 2. BARRIER ANALYSIS FIGURE
# ==============================================================================
print("\nBarrier analysis figure...")

BARRIER_PAP = {
    1:"Never had sex",2:"Not necessary",3:"Embarrassment",
    4:"Not oriented",5:"Doesn't know where",6:"Financial difficulties",
    7:"Long wait time",8:"Distance/transport",9:"Incompatible hours",
    10:"Insurance doesn't cover",11:"Scheduled not done",
    13:"Other",14:"Other"
}
BARRIER_MAMO = {
    1:"Scheduled not done",2:"Not necessary",3:"Didn't know where",
    4:"Couldn't schedule",5:"Financial diff",6:"Transport diff",
    7:"Long wait time",8:"Too far",9:"Incompatible hours",
    10:"Insurance doesn't cover",11:"Other"
}

# Barrier colors per stratum
STRATUM_COLORS = {
    "Low risk":    "#90CAF9",
    "Medium risk": "#42A5F5",
    "High risk":   "#1565C0",
}

def compute_barrier(df_ds, target, age_lim, bar_col, bar_labels, pipe):
    sub = df_ds[(df_ds["age"]>=50) & (df_ds["age"]<=age_lim)].copy()
    sub["age_group"] = pd.cut(sub["age"], bins=[49,59,age_lim],
                               labels=["50-59",f"60-{age_lim}"]).astype(str)
    sub_f = sub[ALL_FEAT].copy().reset_index(drop=True)
    sub_t = pd.to_numeric(sub[target], errors="coerce").values
    sub_b = pd.to_numeric(sub[bar_col], errors="coerce").values

    mask = (sub_t==1) & ~np.isnan(sub_b)
    sub_combined = sub[ALL_FEAT+[target]].copy()
    sub_combined["_barrier"] = sub_b
    sub_combined["_target"]  = sub_t
    sub_na = sub_combined[(sub_combined["_target"]==1) &
                           sub_combined["_barrier"].notna()].copy()
    sub_na = sub_na.dropna(subset=ALL_FEAT)
    if len(sub_na) < 50: return None

    sub_f_na = sub_na[ALL_FEAT].reset_index(drop=True)
    scores   = pipe.predict_proba(sub_f_na)[:,1]
    t33, t67 = np.percentile(scores,[33,67])
    strata   = np.where(scores<t33,"Low risk",
               np.where(scores>t67,"High risk","Medium risk"))
    barriers = sub_na["_barrier"].values

    result = {}
    for stratum in ["Low risk","Medium risk","High risk"]:
        idx = strata==stratum
        bar_s = pd.Series(barriers[idx]).map(bar_labels).dropna()
        prop  = bar_s.value_counts(normalize=True).head(5)
        result[stratum] = {"props":prop, "n":idx.sum()}
    return result

pipe_pap  = joblib.load(MODELS/"model_pap_smear_XGBoost.joblib")
pipe_mamo = joblib.load(MODELS/"model_mammography_XGBoost.joblib")

# Heatmap layout: one heatmap per dataset × outcome combination
# Rows = barriers, Cols = risk tertiles, Color = proportion %
import matplotlib.colors as mcolors

configs_heat = [
    ("Pap smear",   "target_pap",  64, "barrier_pap",  BARRIER_PAP,
     pipe_pap,  [("PNS 2013", pns13,"A"), ("PNS 2019", pns19,"B")]),
    ("Mammography", "target_mamo", 69, "barrier_mamo", BARRIER_MAMO,
     pipe_mamo, [("PNS 2013", pns13,"C")]),
]

# Load barrier data from CSV (same source as Table 3)
bar_csv = OUT_TAB / "barrier_analysis.csv"
if not bar_csv.exists():
    print(f"  ERROR: {bar_csv} not found — run pns_pipeline_v2.py first")
    panels = []
else:
    df_bar = pd.read_csv(bar_csv)
    print(f"  Loaded barrier_analysis.csv: {len(df_bar)} rows")
    print(f"  Columns: {df_bar.columns.tolist()}")

    # Build panels from CSV
    panels = []
    panel_map = [
        ("A", "Pap smear",   "PNS 2013"),
        ("B", "Pap smear",   "PNS 2019"),
        ("C", "Mammography", "PNS 2013"),
    ]
    for pl, outcome, ds_name in panel_map:
        sub = df_bar[(df_bar["Outcome"]==outcome) &
                     (df_bar["Dataset"]==ds_name)].copy()
        if len(sub) == 0:
            print(f"  No data for {outcome} {ds_name}")
            continue
        # Build result dict matching expected structure
        result = {}
        for stratum in ["Low risk","Medium risk","High risk"]:
            s = sub[sub["Stratum"]==stratum].copy()
            if len(s) == 0: continue
            props = pd.Series(
                s["Proportion"].values,
                index=s["Barrier"].values)
            n_val = int(s["n"].iloc[0]) if "n" in s.columns else 0
            result[stratum] = {"props": props, "n": n_val}
        if result:
            panels.append((pl, outcome, ds_name, result))
            print(f"  Panel {pl}: {outcome} / {ds_name} — "
                  f"{len(result)} strata")

n_panels = len(panels)
fig, axes = plt.subplots(1, n_panels,
                          figsize=(5.5 * n_panels, 7))
if n_panels == 1: axes = [axes]
fig.subplots_adjust(wspace=0.5, left=0.02, right=0.98,
                    top=0.90, bottom=0.08)

strata_order = ["Low risk", "Medium risk", "High risk"]
cmap = plt.cm.Blues

for ax, (pl, outcome, ds_name, result) in zip(axes, panels):
    # Build matrix: all barriers × strata
    all_bars = set()
    for s in result.values():
        all_bars.update(s["props"].index.tolist())

    mat = pd.DataFrame(index=sorted(all_bars),
                       columns=strata_order, data=0.0)
    for stratum in strata_order:
        for bar, prop in result[stratum]["props"].items():
            if bar in mat.index:
                mat.loc[bar, stratum] = prop * 100

    # Sort by High risk proportion descending
    mat = mat.sort_values("High risk", ascending=False)

    # Draw heatmap
    data = mat.values
    im = ax.imshow(data, cmap=cmap, aspect="auto",
                   vmin=0, vmax=min(70, data.max()*1.1))

    # Labels
    ax.set_xticks(range(3))
    ax.set_xticklabels(
        [f"{s}\n(n={result[s]['n']:,})" for s in strata_order],
        fontsize=16)
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(mat.index, fontsize=16)
    ax.tick_params(left=False, bottom=False)

    # Annotate cells with percentage
    for i in range(len(mat)):
        for j in range(3):
            val = data[i, j]
            color = "white" if val > 35 else "black"
            ax.text(j, i, f"{val:.1f}%",
                    ha="center", va="center",
                    fontsize=14, color=color, fontweight="bold")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Proportion (%)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title(f"{pl}  {outcome} — {ds_name}",
                 fontsize=FONT_TITLE, fontweight="bold",
                 loc="left", pad=10)

    fig.suptitle("Reported barriers to cancer screening non-adherence "
             "by predicted risk tertile (XGBoost, PNS external validation)",
             fontsize=12, fontweight="bold", y=0.98)

out = OUT_FIG/"fig_barrier_analysis.pdf"
fig.savefig(out, format="pdf", bbox_inches="tight",
            facecolor="white", dpi=300)
plt.close()
print(f"  Saved: {out.name}")

# ==============================================================================
# 3. DECISION CURVE ANALYSIS (DCA)
# ==============================================================================
print("\nDecision curve analysis...")

def net_benefit(y, yp, threshold):
    n = len(y)
    yc = (yp >= threshold).astype(int)
    tp = ((yc==1) & (y==1)).sum()
    fp = ((yc==1) & (y==0)).sum()
    return (tp/n) - (fp/n) * (threshold/(1-threshold+1e-10))

def dca_curve(y, yp, thresholds):
    return np.array([net_benefit(y, yp, t) for t in thresholds])

thresholds = np.linspace(0.01, 0.70, 100)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.subplots_adjust(wspace=0.3, left=0.07, right=0.97,
                    top=0.90, bottom=0.10)

PALETTE_DCA = {
    "ELSI W3\n(temporal val.)":           "#2E7D32",
    "PNS 2013\n(ext. retro val.)":        "#E65100",
    "PNS 2019\n(ext. prosp val.)":        "#6A1B9A",
}
LS_DCA = {
    "ELSI W3\n(temporal val.)":    "--",
    "PNS 2013\n(ext. retro val.)": "-.",
    "PNS 2019\n(ext. prosp val.)": ":",
}

for col_i, (outcome, target, age_elsi, age_pns, panel_label) in enumerate([
    ("Pap smear",   "target_pap",  64, 64, "A"),
    ("Mammography", "target_mamo", 69, 69, "B"),
]):
    ax   = axes[col_i]
    pipe = joblib.load(MODELS/f"model_{outcome.lower().replace(' ','_')}_XGBoost.joblib")

    datasets_dca = [
        ("ELSI W3\n(temporal val.)",    w3,    age_elsi),
        ("PNS 2013\n(ext. retro val.)", pns13, age_pns),
        ("PNS 2019\n(ext. prosp val.)", pns19, age_pns),
    ]

    for ds_label, df_ds, age_lim in datasets_dca:
        X, y = get_Xy(df_ds, target, 50, age_lim)
        yp   = pipe.predict_proba(X)[:,1]
        prev = y.mean()

        nb   = dca_curve(y, yp, thresholds)
        # Treat all as positive
        nb_all = np.array([(prev - (1-prev)*(t/(1-t+1e-10)))
                            for t in thresholds])

        ax.plot(thresholds*100, nb, color=PALETTE_DCA[ds_label],
                lw=1.8, linestyle=LS_DCA[ds_label],
                label=ds_label.replace("\n"," "))

    # Reference lines
    X0, y0 = get_Xy(w3, target, 50, age_elsi)
    prev0   = y0.mean()
    nb_all0 = np.array([(prev0 - (1-prev0)*(t/(1-t+1e-10)))
                         for t in thresholds])
    ax.plot(thresholds*100, nb_all0, color="grey", lw=1.2,
            linestyle="-", alpha=0.6, label="Treat all")
    ax.axhline(0, color="black", lw=0.8, linestyle="-",
               alpha=0.5, label="Treat none")

    ax.set_xlim(0, 65)
    ax.set_ylim(-0.05, max(prev0*1.2, 0.15))
    ax.set_xlabel("Threshold probability (%)", fontsize=FONT_LABEL)
    ax.set_ylabel("Net benefit", fontsize=FONT_LABEL)
    ax.set_title(f"{panel_label}  {outcome}",
                 fontsize=FONT_TITLE, fontweight="bold", loc="left")
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.9, edgecolor="grey")

fig.suptitle("Decision curve analysis — XGBoost non-adherence prediction model",
             fontsize=12, fontweight="bold", y=0.98)
fig.text(0.5, 0.01,
    "Net benefit = TP rate \u2212 FP rate \u00d7 (threshold / (1 \u2212 threshold)). "
    "'Treat all': screen all women regardless of risk. "
    "'Treat none': screen no women.",
    ha="center", fontsize=7.5, color="grey", style="italic")

out = OUT_FIG/"fig_dca.pdf"
fig.savefig(out, format="pdf", bbox_inches="tight",
            facecolor="white", dpi=300)
plt.close()
print(f"  Saved: {out.name}")

print("\n=== ALL FIGURES SAVED ===")
print(f"  {OUT_FIG}/fig_shap_pap_smear.pdf")
print(f"  {OUT_FIG}/fig_shap_mammography.pdf")
print(f"  {OUT_FIG}/fig_barrier_analysis.pdf")
print(f"  {OUT_FIG}/fig_dca.pdf")
