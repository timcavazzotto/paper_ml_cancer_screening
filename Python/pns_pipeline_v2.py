"""
Cancer Screening Non-Adherence — Pipeline v2
=============================================
Design:
  Train:               ELSI W1 (2015-16) + ELSI W2 (2019-21) [pooled]
  Temporal validation: ELSI W3 (2023-24)
  External validation: PNS 2013 (retrospective) + PNS 2019 (prospective)
  Barrier analysis:    PNS 2013 (Pap + Mamo) + PNS 2019 (Pap only)
  Ecological:          PAINEL-Oncologia (separate R script)

Folder structure:
  ROOT = Final_5_samples/
    Data/
    Codes/Python/
    Codes/R/
    Outputs/Python_figures/
    Outputs/Python_tables/
    Outputs/R_figures/
    Outputs/R_tables/
    Manuscript/
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, pyreadstat
from pathlib import Path

from sklearn.pipeline        import Pipeline
from sklearn.compose         import ColumnTransformer
from sklearn.preprocessing   import OneHotEncoder
from sklearn.impute          import SimpleImputer
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.metrics         import (roc_auc_score, brier_score_loss,
                                     f1_score, roc_curve, RocCurveDisplay)
from xgboost import XGBClassifier
import shap
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ==============================================================================
# FOLDER STRUCTURE
# ==============================================================================
ROOT      = Path(__file__).parent.parent.parent  # Final_5_samples/
DATA      = ROOT / "Data"
OUT_PY_FIG= ROOT / "Outputs" / "Python_figures"
OUT_PY_TAB= ROOT / "Outputs" / "Python_tables"
MODELS    = ROOT / "Outputs" / "Models"   # saved .joblib files

OUT_R = ROOT / "Codes" / "R"
for d in [DATA, OUT_PY_FIG, OUT_PY_TAB, MODELS, OUT_R]:
    d.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# DATA PATHS
# ==============================================================================
ELSI_W1 = DATA / "ELSI Portugues (1a onda) stata13.dta"
ELSI_W2 = DATA / "ELSI Portugues (2a onda) stata13.dta"
ELSI_W3 = DATA / "ELSI Portugues (3a onda).dta"
PNS13   = DATA / "PNS_2013.csv"
PNS19   = DATA / "pns2019.csv"

# ==============================================================================
# CONSTANTS
# ==============================================================================
MISS = {9, 99, 999, 88, 8}
def clean(s): return pd.to_numeric(s, errors="coerce").replace(list(MISS), np.nan)

ETHNIC_ELSI = {1:"White", 2:"Black", 3:"Brown", 4:"Yellow", 5:"Indigenous"}
ETHNIC_PNS  = {1:"White", 2:"Black", 3:"Yellow", 4:"Brown",  5:"Indigenous"}

MAR_ELSI  = {1:"Without partner", 2:"With partner",
             3:"Without partner", 4:"Without partner"}
MAR_PNS13 = {1:"With partner",    2:"Without partner",
             3:"Without partner", 4:"Without partner", 5:"Without partner"}
MAR_PNS19 = {1:"With partner",    2:"Without partner",
             3:"Without partner", 4:"Without partner"}

INS_ELSI = {0:"No", 1:"Yes"}   # u1: 0=Nao, 1=Sim
INS_PNS  = {1:"Yes", 2:"No"}

ZONE_R = {1:"Urban", 2:"Rural"}

def educ_elsi(v):
    if pd.isna(v): return np.nan
    v = int(v)
    if v == 1:        return "Illiterate"
    if 2 <= v <= 8:   return "Incomplete Elem"
    if 9 <= v <= 13:  return "Complete Elem"
    if 14 <= v <= 18: return "Higher Educ"
    return np.nan

EDUC_PNS = {1:"Illiterate", 2:"Incomplete Elem", 3:"Complete Elem",
            4:"Incomplete Elem", 5:"Complete Elem",
            6:"Higher Educ",    7:"Higher Educ"}

UF_REG = {
    11:"North",12:"North",13:"North",14:"North",15:"North",16:"North",17:"North",
    21:"Northeast",22:"Northeast",23:"Northeast",24:"Northeast",25:"Northeast",
    26:"Northeast",27:"Northeast",28:"Northeast",29:"Northeast",
    31:"Southeast",32:"Southeast",33:"Southeast",35:"Southeast",
    41:"South",42:"South",43:"South",
    50:"Central-West",51:"Central-West",52:"Central-West",53:"Central-West",
}
REG_ELSI = {1:"North",2:"Northeast",3:"Southeast",4:"South",5:"Central-West"}

CAT_FEAT = ["age_group","region","zone","marital","ethnicity",
            "education","insurance"]
NUM_FEAT = ["n_residents"]
ALL_FEAT = CAT_FEAT + NUM_FEAT

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ==============================================================================
# DATA PREPARATION — ELSI
# ==============================================================================
def prep_elsi(path, wave):
    print(f"  Loading ELSI W{wave}...")
    df, _ = pyreadstat.read_dta(str(path), encoding="latin1")
    o = pd.DataFrame()

    o["age"]       = clean(df["idade"])
    o["sex"]       = pd.to_numeric(df["sexo"], errors="coerce")
    o["ethnicity"] = clean(df["e9"]).map(ETHNIC_ELSI)
    o["marital"]   = clean(df["e7"]).map(MAR_ELSI)
    o["zone"]      = clean(df["zona"]).map(ZONE_R)
    o["region"]    = clean(df["regiao"]).map(REG_ELSI)
    o["education"] = clean(df["e22"]).apply(educ_elsi)
    o["insurance"] = clean(df["u1"]).map(INS_ELSI)

    res_col = "ar6" if wave == 1 else "nmoradores"
    o["n_residents"] = clean(df[res_col]).clip(upper=10)

    # Pap target — identical across waves
    m13 = clean(df["m13"])
    o["target_pap"] = m13.isin({2,3}).astype("Int64")
    o["target_pap"] = o["target_pap"].where(m13.isin({1,2,3}), np.nan)

    # Mamo target — W3 has inverted coding
    m15 = clean(df["m15"])
    if wave in [1, 2]:
        # 1=Never, 2=<1y, 3=1-2y, 4=2-3y, 5=>=3y → non-adherent={1,5}
        o["target_mamo"] = m15.isin({1,5}).astype("Int64")
        o["target_mamo"] = o["target_mamo"].where(m15.isin({1,2,3,4,5}), np.nan)
    else:
        # W3: 1=<1y, 2=1-2y, 3=2-3y, 4=>=3y, 5=Never → non-adherent={4,5}
        o["target_mamo"] = m15.isin({4,5}).astype("Int64")
        o["target_mamo"] = o["target_mamo"].where(m15.isin({1,2,3,4,5}), np.nan)

    o["wave"] = wave
    o = o[(o["sex"]==0) & (o["age"]>=50)].copy()
    print(f"    Women >= 50: {len(o):,}")
    return o

# ==============================================================================
# DATA PREPARATION — PNS
# ==============================================================================
def prep_pns(path, sep, year):
    print(f"  Loading PNS {year}...")
    df = pd.read_csv(path, sep=sep, encoding="latin1", low_memory=False)
    o  = pd.DataFrame()

    o["age"]       = clean(df["C008"])
    o["sex"]       = clean(df["C006"])
    o["ethnicity"] = clean(df["C009"]).map(ETHNIC_PNS)
    o["zone"]      = clean(df["V0026"]).map(ZONE_R)
    o["region"]    = pd.to_numeric(df["V0001"], errors="coerce").map(UF_REG)
    o["education"] = clean(df["VDD004A"]).map(EDUC_PNS)
    o["n_residents"]= clean(df["V0022"]).clip(upper=10)

    if year == 2013:
        o["marital"]  = clean(df["C011"]).map(MAR_PNS13)
        o["insurance"]= clean(df["I001"]).map(INS_PNS)
        pap = clean(df["R001"])
        o["target_pap"] = pap.isin([4,5]).astype("Int64")
        o["target_pap"] = o["target_pap"].where(pap.isin([1,2,3,4,5]), np.nan)
        # Exclude hysterectomised women (R002==12) from Pap sample
        r010 = pd.to_numeric(df["R010"], errors="coerce")
        o["target_pap"] = o["target_pap"].where(r010 != 1, np.nan)
        md = clean(df["R015"]); mw = clean(df["R017"])
        o["target_mamo"] = ((md==2)|mw.isin([3,4])).astype("Int64")
        o["target_mamo"] = o["target_mamo"].where(
            md.isin([1,2])|mw.isin([1,2,3,4]), np.nan)
        o["barrier_pap"]  = pd.to_numeric(df["R002"],    errors="coerce")
        o["barrier_mamo"] = pd.to_numeric(df["R016"],    errors="coerce")
    else:
        o["marital"]  = clean(df["C011"]).map(MAR_PNS19)
        o["insurance"]= clean(df["I00102"]).map(INS_PNS)
        pap = clean(df["R00101"])
        o["target_pap"] = pap.isin([4,5]).astype("Int64")
        o["target_pap"] = o["target_pap"].where(pap.isin([1,2,3,4,5]), np.nan)
        # Exclude hysterectomised women (R002010==12) from Pap sample
        r002010 = pd.to_numeric(df["R002010"], errors="coerce")
        o["target_pap"] = o["target_pap"].where(r002010 != 12, np.nan)
        md = clean(df["R015"]); mw = clean(df["R01701"])
        o["target_mamo"] = ((md==2)|mw.isin([3,4])).astype("Int64")
        o["target_mamo"] = o["target_mamo"].where(
            md.isin([1,2])|mw.isin([1,2,3,4]), np.nan)
        o["barrier_pap"]  = pd.to_numeric(df["R002010"], errors="coerce")
        o["barrier_mamo"] = np.nan

    o["year"] = year
    o = o[(o["sex"]==2) & (o["age"]>=50)].copy()
    print(f"    Women >= 50: {len(o):,}")
    return o

# ==============================================================================
# HELPERS
# ==============================================================================
def get_Xy(df, target, age_min, age_max):
    s = df[(df["age"]>=age_min) & (df["age"]<=age_max)].copy()
    s["age_group"] = pd.cut(s["age"], bins=[age_min-1,59,age_max],
                            labels=["50-59",f"60-{age_max}"]).astype(str)
    sub = s[ALL_FEAT+[target]].dropna()
    return sub[ALL_FEAT].reset_index(drop=True), sub[target].astype(int).values

def make_preprocessor():
    cat = Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                    ("ohe",OneHotEncoder(handle_unknown="ignore",
                                        sparse_output=False))])
    num = Pipeline([("imp",SimpleImputer(strategy="median"))])
    return ColumnTransformer([("cat",cat,CAT_FEAT),
                               ("num",num,NUM_FEAT)], remainder="drop")

def get_feat_names(pipe):
    ohe = pipe["pre"].named_transformers_["cat"]["ohe"]
    return ohe.get_feature_names_out(CAT_FEAT).tolist() + NUM_FEAT

def youden_thr(y, yp):
    fpr,tpr,thr = roc_curve(y,yp)
    return float(thr[np.argmax(tpr-fpr)])

def compute_metrics(pipe, X, y, thr=0.50):
    from sklearn.metrics import confusion_matrix
    yp = pipe.predict_proba(X)[:,1]
    yc = (yp>=thr).astype(int)
    tn,fp,fn,tp = confusion_matrix(y,yc).ravel()
    return {
        "AUC":   round(roc_auc_score(y,yp),3),
        "Brier": round(brier_score_loss(y,yp),3),
        "Sens":  round(tp/(tp+fn),3) if (tp+fn)>0 else np.nan,
        "Spec":  round(tn/(tn+fp),3) if (tn+fp)>0 else np.nan,
        "PPV":   round(tp/(tp+fp),3) if (tp+fp)>0 else np.nan,
        "NPV":   round(tn/(tn+fn),3) if (tn+fn)>0 else np.nan,
    }

ALGO = {
    "LogReg": LogisticRegression(penalty="elasticnet",solver="saga",
                                  max_iter=3000,random_state=42,
                                  class_weight="balanced"),
    "RandomForest": RandomForestClassifier(random_state=42,
                                            class_weight="balanced",n_jobs=-1),
    "XGBoost": XGBClassifier(eval_metric="logloss",random_state=42,
                               use_label_encoder=False,n_jobs=-1),
}
GRIDS = {
    "LogReg":       {"clf__C":[0.001,0.01,0.1,1,10],
                     "clf__l1_ratio":[0.0,0.25,0.5,0.75,1.0]},
    "RandomForest": {"clf__n_estimators":[300,500],
                     "clf__max_depth":[None,6,10],
                     "clf__min_samples_split":[2,4,8]},
    "XGBoost":      {"clf__n_estimators":[300,500],
                     "clf__max_depth":[3,4,5],
                     "clf__learning_rate":[0.01,0.05,0.1],
                     "clf__subsample":[0.7,0.85,1.0],
                     "clf__colsample_bytree":[0.6,0.8,1.0]},
}

# ==============================================================================
# RUN ONE OUTCOME
# ==============================================================================
def run_outcome(outcome, target, age_elsi, age_pns,
                train_data, w3, pns13, pns19):
    print(f"\n{'='*60}\nOUTCOME: {outcome.upper()}\n{'='*60}")

    X_tr,  y_tr  = get_Xy(train_data, target, 50, age_elsi)
    X_w3,  y_w3  = get_Xy(w3,         target, 50, age_elsi)
    X_p13, y_p13 = get_Xy(pns13,      target, 50, age_pns)
    X_p19, y_p19 = get_Xy(pns19,      target, 50, age_pns)

    print(f"  Train (ELSI W1+W2):      n={len(y_tr):,}  pos={y_tr.mean():.1%}")
    print(f"  Temporal val (ELSI W3):  n={len(y_w3):,}  pos={y_w3.mean():.1%}")
    print(f"  Ext val PNS 2013 (retro):n={len(y_p13):,}  pos={y_p13.mean():.1%}")
    print(f"  Ext val PNS 2019 (prosp):n={len(y_p19):,}  pos={y_p19.mean():.1%}")

    pre = make_preprocessor()
    results = {}; pipes = {}

    print("\n  Tuning models:")
    for name, clf in ALGO.items():
        print(f"    {name}...", end=" ", flush=True)
        if name == "XGBoost":
            clf = clf.set_params(
                scale_pos_weight=(y_tr==0).sum()/max((y_tr==1).sum(),1))
        pipe = Pipeline([("pre",pre),("clf",clf)])
        rs = RandomizedSearchCV(pipe,GRIDS[name],n_iter=40,scoring="roc_auc",
                                cv=CV,refit=True,n_jobs=-1,random_state=42)
        rs.fit(X_tr,y_tr)
        best = rs.best_estimator_

        cv_r = cross_validate(best,X_tr,y_tr,cv=CV,
                              scoring={"auc":"roc_auc"},n_jobs=-1)

        # Youden threshold from temporal validation (W3)
        thr = youden_thr(y_w3, best.predict_proba(X_w3)[:,1])

        m = {"AUC_CV": cv_r["test_auc"].mean(),
             "AUC_CV_std": cv_r["test_auc"].std()}

        for label,X,y in [("W3",X_w3,y_w3),
                           ("PNS13",X_p13,y_p13),
                           ("PNS19",X_p19,y_p19)]:
            mm = compute_metrics(best,X,y,thr)
            m.update({f"{k}_{label}":v for k,v in mm.items()})

        print(f"AUC_CV={m['AUC_CV']:.3f}±{m['AUC_CV_std']:.3f} "
              f"| W3={m['AUC_W3']:.3f} "
              f"| PNS13={m['AUC_PNS13']:.3f} "
              f"| PNS19={m['AUC_PNS19']:.3f}")

        results[name] = m
        pipes[name]   = best
        joblib.dump(best, MODELS/f"model_{outcome}_{name}.joblib")

    # SHAP — XGBoost on W3
    print(f"    SHAP (W3)...", end=" ", flush=True)
    fn    = get_feat_names(pipes["XGBoost"])
    Xtr_t = pd.DataFrame(pipes["XGBoost"]["pre"].transform(X_tr), columns=fn)
    Xw3_t = pd.DataFrame(pipes["XGBoost"]["pre"].transform(X_w3), columns=fn)
    expl  = shap.TreeExplainer(pipes["XGBoost"]["clf"])
    sv    = expl.shap_values(Xw3_t)
    if isinstance(sv,list): sv = sv[1]
    if sv.ndim==3: sv = sv[:,:,1]
    plt.figure(figsize=(11,8))
    shap.summary_plot(sv,Xw3_t,max_display=15,show=False)
    plt.title(f"SHAP — {outcome} (XGBoost, ELSI W3 validation)",fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_PY_FIG/f"shap_{outcome}.png",dpi=300,bbox_inches="tight")
    plt.savefig(OUT_PY_FIG/f"shap_{outcome}.pdf",bbox_inches="tight")
    plt.close()
    pd.DataFrame(sv,columns=fn).to_csv(
        OUT_PY_TAB/f"shap_values_{outcome}.csv",index=False)
    print("done")

    return results, pipes

# ==============================================================================
# BARRIER ANALYSIS
# ==============================================================================
BARRIER_PAP = {
    1:"Never had sex",    2:"Not necessary",      3:"Embarrassment",
    4:"Not oriented",     5:"Doesn't know where", 6:"Financial difficulties",
    7:"Long wait time",   8:"Distance/transport", 9:"Incompatible hours",
    10:"Insurance doesn't cover", 11:"Scheduled not done",
    12:"Hysterectomy",   13:"Other", 14:"Other"
}
# Note: barrier code 12 (Hysterectomy) excluded from target — women who
# reported hysterectomy are set as missing in target_pap (not non-adherent)
BARRIER_MAMO = {
    1:"Scheduled not done", 2:"Not necessary",    3:"Didn't know where",
    4:"Couldn't schedule",  5:"Financial diff",   6:"Transport diff",
    7:"Long wait time",     8:"Too far",           9:"Incompatible hours",
    10:"Insurance doesn't cover", 11:"Other"
}

def barrier_analysis(pns13, pns19, pipes_pap, pipes_mamo):
    print("\n" + "="*60)
    print("BARRIER ANALYSIS")
    print("="*60)

    results = []
    for ds_name, df_ds, has_mamo in [
        ("PNS 2013", pns13, True),
        ("PNS 2019", pns19, False),
    ]:
        for outcome, target, age_lim, bar_col, bar_labels, pipe in [
            ("Pap smear",   "target_pap",  64,
             "barrier_pap",  BARRIER_PAP,  pipes_pap["XGBoost"]),
            ("Mammography", "target_mamo", 69,
             "barrier_mamo", BARRIER_MAMO, pipes_mamo["XGBoost"]),
        ]:
            if outcome=="Mammography" and not has_mamo: continue

            sub = df_ds[(df_ds["age"]>=50)&(df_ds["age"]<=age_lim)].copy()
            sub["age_group"] = pd.cut(sub["age"],bins=[49,59,age_lim],
                                      labels=["50-59",f"60-{age_lim}"]).astype(str)
            sub_f = sub[ALL_FEAT].reset_index(drop=True)
            sub_t = pd.to_numeric(sub[target],  errors="coerce").values
            sub_b = pd.to_numeric(sub[bar_col], errors="coerce").values

            # Non-adherent with barrier info and complete predictors
            # Build a combined dataframe to ensure alignment
            sub_combined = sub_f.copy()
            sub_combined["_target"]  = sub_t
            sub_combined["_barrier"] = sub_b

            # Filter: non-adherent with barrier info
            sub_na = sub_combined[
                (sub_combined["_target"]==1) &
                sub_combined["_barrier"].notna()
            ].copy()

            # Drop rows with missing predictors
            sub_na = sub_na.dropna(subset=ALL_FEAT)

            if len(sub_na) < 50:
                print(f"  {ds_name} {outcome}: n too small for barrier analysis ({len(sub_na)})")
                continue

            sub_f_na = sub_na[ALL_FEAT].reset_index(drop=True)
            sub_b_na = sub_na["_barrier"].values

            if len(sub_f_na) < 50:
                print(f"  {ds_name} {outcome}: n too small for barrier analysis")
                continue

            scores  = pipe.predict_proba(sub_f_na)[:,1]
            t33,t67 = np.percentile(scores,[33,67])
            strata  = np.where(scores<t33,"Low risk",
                      np.where(scores>t67,"High risk","Medium risk"))

            df_bar = pd.DataFrame({"barrier":sub_b_na,"stratum":strata})
            df_bar["barrier_label"] = df_bar["barrier"].map(bar_labels)

            for stratum in ["Low risk","Medium risk","High risk"]:
                s = df_bar[df_bar["stratum"]==stratum]
                if len(s)==0: continue
                top = (s["barrier_label"].value_counts(normalize=True)
                       .head(5).reset_index())
                top.columns = ["Barrier","Proportion"]
                top["Dataset"]=ds_name; top["Outcome"]=outcome
                top["Stratum"]=stratum; top["n"]=len(s)
                results.append(top)

            print(f"  {ds_name} {outcome}: n_non_adh={len(sub_f_na):,} | "
                  f"Low={(strata=='Low risk').sum()} "
                  f"Med={(strata=='Medium risk').sum()} "
                  f"High={(strata=='High risk').sum()}")

    if results:
        df_out = pd.concat(results,ignore_index=True)
        df_out.to_csv(OUT_PY_TAB/"barrier_analysis.csv",index=False)
        print(f"\n  Saved: Outputs/Python_tables/barrier_analysis.csv")

# ==============================================================================
# SUMMARY TABLE
# ==============================================================================
def summary_table(all_results):
    rows = []
    for outcome, models in all_results.items():
        for mname, m in models.items():
            rows.append({
                "Outcome": outcome, "Model": mname,
                "AUC_CV":    f"{m['AUC_CV']:.3f}±{m['AUC_CV_std']:.3f}",
                "AUC_W3":    f"{m['AUC_W3']:.3f}",
                "AUC_PNS13": f"{m['AUC_PNS13']:.3f}",
                "AUC_PNS19": f"{m['AUC_PNS19']:.3f}",
                "Sens_W3":   f"{m['Sens_W3']:.3f}",
                "Spec_W3":   f"{m['Spec_W3']:.3f}",
                "Brier_W3":  f"{m['Brier_W3']:.3f}",
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_PY_TAB/"performance_summary.csv",index=False)
    print("\n=== PERFORMANCE SUMMARY ===")
    print(df.to_string(index=False))
    return df


# ==============================================================================
# DOCX HELPERS
# ==============================================================================
def _set_bg(cell, color):
    tc=cell._tc; pr=tc.get_or_add_tcPr()
    shd=OxmlElement("w:shd")
    shd.set(qn("w:val"),"clear"); shd.set(qn("w:color"),"auto")
    shd.set(qn("w:fill"),color); pr.append(shd)

def _set_txt(cell, text, bold=False, size=8, align=WD_ALIGN_PARAGRAPH.CENTER):
    cell.text=""
    p=cell.paragraphs[0]; p.alignment=align
    r=p.add_run(str(text)); r.bold=bold
    r.font.size=Pt(size); r.font.name="Times New Roman"

def _footnote(doc, text):
    p=doc.add_paragraph()
    r=p.add_run(text); r.font.size=Pt(8); r.font.name="Times New Roman"
    p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY

def _new_doc():
    doc=Document()
    for s in doc.sections:
        s.top_margin=Cm(2); s.bottom_margin=Cm(2)
        s.left_margin=Cm(2); s.right_margin=Cm(2)
    return doc

# ==============================================================================
# BOOTSTRAP CI (Table 2)
# ==============================================================================
N_BOOT = 1000

def _youden(y,yp):
    from sklearn.metrics import roc_curve
    fpr,tpr,thr = roc_curve(y,yp)
    return float(thr[np.argmax(tpr-fpr)])

def _point(y,yp,thr):
    from sklearn.metrics import confusion_matrix
    yc=(yp>=thr).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,yc).ravel()
    return dict(
        AUC=round(roc_auc_score(y,yp),3),
        Brier=round(brier_score_loss(y,yp),3),
        Sens=round(tp/(tp+fn),3) if (tp+fn)>0 else np.nan,
        Spec=round(tn/(tn+fp),3) if (tn+fp)>0 else np.nan,
        PPV=round(tp/(tp+fp),3) if (tp+fp)>0 else np.nan,
        NPV=round(tn/(tn+fn),3) if (tn+fn)>0 else np.nan,
    )

def _boot(y,yp,thr,n=N_BOOT,seed=42):
    rng=np.random.default_rng(seed)
    y=np.asarray(y); yp=np.asarray(yp)
    acc={k:[] for k in ["AUC","Brier","Sens","Spec","PPV","NPV"]}
    for _ in range(n):
        idx=rng.integers(0,len(y),size=len(y))
        yb,ypb=y[idx],yp[idx]
        if len(np.unique(yb))<2: continue
        for k,v in _point(yb,ypb,thr).items():
            if not np.isnan(v): acc[k].append(v)
    ci={}
    for k,vals in acc.items():
        if vals:
            ci[f"{k}_lo"]=round(np.percentile(vals,2.5),3)
            ci[f"{k}_hi"]=round(np.percentile(vals,97.5),3)
        else: ci[f"{k}_lo"]=ci[f"{k}_hi"]=np.nan
    return ci

def _fmt(p,lo,hi): return f"{p:.3f} ({lo:.3f}\u2013{hi:.3f})"

def bootstrap_ci(w3, pns13, pns19, pipes_pap, pipes_mamo):
    print("\n" + "="*60)
    print("BOOTSTRAP CI (Table 2, n=1000 resamples)")
    print("="*60)
    rows=[]
    for outcome,target,age_elsi,age_pns,pipes in [
        ("Pap smear",  "target_pap",  64,64,pipes_pap),
        ("Mammography","target_mamo", 69,69,pipes_mamo),
    ]:
        pipe=pipes["XGBoost"]
        X_w3,y_w3=get_Xy(w3,target,50,age_elsi)
        thr=_youden(y_w3,pipe.predict_proba(X_w3)[:,1])
        print(f"  {outcome}: Youden thr={thr:.3f}")

        for ds_label,df_ds,age_lim in [
            ("ELSI W3 (temporal)",   w3,   age_elsi),
            ("PNS 2013 (ext.retro)", pns13,age_pns),
            ("PNS 2019 (ext.prosp)", pns19,age_pns),
        ]:
            X,y=get_Xy(df_ds,target,50,age_lim)
            yp=pipe.predict_proba(X)[:,1]
            m=_point(y,yp,thr); ci=_boot(y,yp,thr)
            print(f"    {ds_label}: AUC={m['AUC']:.3f} Sens={m['Sens']:.3f}",
                  end="... boot...", flush=True)
            rows.append({
                "Outcome":outcome,"Dataset":ds_label,"n":len(y),
                "Prev (%)":f"{y.mean()*100:.1f}",
                "AUC (95% CI)":  _fmt(m["AUC"],  ci["AUC_lo"],  ci["AUC_hi"]),
                "Brier (95% CI)":_fmt(m["Brier"],ci["Brier_lo"],ci["Brier_hi"]),
                "Sens (95% CI)": _fmt(m["Sens"], ci["Sens_lo"], ci["Sens_hi"]),
                "Spec (95% CI)": _fmt(m["Spec"], ci["Spec_lo"], ci["Spec_hi"]),
                "PPV (95% CI)":  _fmt(m["PPV"],  ci["PPV_lo"],  ci["PPV_hi"]),
                "NPV (95% CI)":  _fmt(m["NPV"],  ci["NPV_lo"],  ci["NPV_hi"]),
            })
            print("done")

    df_t2=pd.DataFrame(rows)
    df_t2.to_csv(OUT_PY_TAB/"table2_performance_ci.csv",index=False)

    # Word
    doc=_new_doc()
    doc.add_paragraph().add_run(
        "Table 2. Predictive performance of XGBoost models across validation "
        "datasets (95% bootstrap CI, n=1,000 resamples).").bold=True
    doc.add_paragraph()
    COLS=[
        ("Outcome","Outcome",Cm(2.5),WD_ALIGN_PARAGRAPH.LEFT),
        ("Dataset","Dataset",Cm(2.8),WD_ALIGN_PARAGRAPH.LEFT),
        ("n","n",Cm(1.2),WD_ALIGN_PARAGRAPH.CENTER),
        ("Prev\n(%)","Prev (%)",Cm(1.0),WD_ALIGN_PARAGRAPH.CENTER),
        ("AUC\n(95% CI)","AUC (95% CI)",Cm(3.2),WD_ALIGN_PARAGRAPH.CENTER),
        ("Brier\n(95% CI)","Brier (95% CI)",Cm(3.0),WD_ALIGN_PARAGRAPH.CENTER),
        ("Sens\n(95% CI)","Sens (95% CI)",Cm(3.0),WD_ALIGN_PARAGRAPH.CENTER),
        ("Spec\n(95% CI)","Spec (95% CI)",Cm(3.0),WD_ALIGN_PARAGRAPH.CENTER),
        ("PPV\n(95% CI)","PPV (95% CI)",Cm(3.0),WD_ALIGN_PARAGRAPH.CENTER),
        ("NPV\n(95% CI)","NPV (95% CI)",Cm(3.0),WD_ALIGN_PARAGRAPH.CENTER),
    ]
    t=doc.add_table(rows=1+len(rows),cols=len(COLS)); t.style="Table Grid"
    for ci2,(_,_,w,_) in enumerate(COLS):
        for cell in t.columns[ci2].cells: cell.width=w
    hdr=t.rows[0].cells
    for ci2,(h,_,_,a) in enumerate(COLS):
        _set_txt(hdr[ci2],h,bold=True,size=8); _set_bg(hdr[ci2],"D9D9D9")
    prev_out=None
    for ri,rd in enumerate(rows,1):
        row=t.rows[ri].cells
        bg="F5F5F5" if rd["Outcome"]=="Pap smear" else "FFFFFF"
        for ci2,(_,key,_,a) in enumerate(COLS):
            val=rd.get(key,"")
            if key=="Outcome":
                val=val if val!=prev_out else ""; prev_out=rd["Outcome"]
            _set_txt(row[ci2],str(val),size=8,align=a); _set_bg(row[ci2],bg)
    doc.add_paragraph()
    _footnote(doc,"AUC: area under the ROC curve. Brier: Brier score. "
              "Sens: sensitivity. Spec: specificity. PPV/NPV: predictive values. "
              "Youden threshold estimated on ELSI W3 and applied to all datasets. "
              "XGBoost trained on ELSI W1+W2 pool.")
    doc.save(OUT_PY_TAB/"table2_performance_ci.docx")
    print("  Saved: table2_performance_ci.csv + .docx")

# ==============================================================================
# EXPORT RISK SCORES BY UF
# ==============================================================================
UF_SIGLA = {
    11:"RO",12:"AC",13:"AM",14:"RR",15:"PA",16:"AP",17:"TO",
    21:"MA",22:"PI",23:"CE",24:"RN",25:"PB",26:"PE",27:"AL",
    28:"SE",29:"BA",31:"MG",32:"ES",33:"RJ",35:"SP",
    41:"PR",42:"SC",43:"RS",50:"MS",51:"MT",52:"GO",53:"DF"
}

def export_risk_scores(pns19, pipes_pap, pipes_mamo):
    print("\n" + "="*60)
    print("EXPORTING RISK SCORES BY UF (for ecological analysis)")
    print("="*60)

    df=pns19.copy()
    pipe_pap  = pipes_pap["XGBoost"]
    pipe_mamo = pipes_mamo["XGBoost"]

    results=[]
    for uf_code, grp in df.groupby(df["region"].map(
            {v:k for k,v in UF_SIGLA.items()}).fillna(-1)):
        pass

    # Use V0001 directly from pns19 — need UF code
    # pns19 already has region as string, need numeric UF
    # Re-read UF from the stored data is complex — store UF in pns19
    # For now group by region (5 groups) as fallback
    # Better: add uf column to prep_pns output
    print("  Note: grouping by region (5 groups) — add UF to prep_pns for state-level")

    for region, grp in df.groupby("region"):
        row={"region":region}
        for outcome,pipe,age_lim,col,ncol in [
            ("pap", pipe_pap,  64,"risk_pap","n_pap"),
            ("mamo",pipe_mamo, 69,"risk_mamo","n_mamo"),
        ]:
            g=grp[grp["age"]<=age_lim].copy()
            g["age_group"]=pd.cut(g["age"],bins=[49,59,age_lim],
                                  labels=["50-59",f"60-{age_lim}"]).astype(str)
            g=g.dropna(subset=ALL_FEAT)
            if len(g)>=30:
                row[col]=pipe.predict_proba(g[ALL_FEAT])[:,1].mean()
                row[ncol]=len(g)
            else:
                row[col]=np.nan; row[ncol]=len(g)
        results.append(row)

    df_out=pd.DataFrame(results)
    df_out.to_csv(OUT_PY_TAB/"risk_scores_region.csv",index=False)
    print(f"  Saved: risk_scores_region.csv")
    print(df_out.to_string(index=False))
    print("  Note: run export_risk_scores_v2.py for UF-level scores (requires UF column)")

# ==============================================================================
# EXPORT NOMOGRAM DATA + R SCRIPTS
# ==============================================================================
def export_nomogram_data(train, pipes_pap, pipes_mamo):
    print("\n" + "="*60)
    print("EXPORTING NOMOGRAM DATA + R SCRIPTS")
    print("="*60)

    for outcome,target,age_min,age_max,label,guideline,cid in [
        ("Pap smear",  "target_pap",  50,64,"Pap_smear",
         "INCA 2016 (50-64y)","C53"),
        ("Mammography","target_mamo", 50,69,"Mammography",
         "MS/CFM 2023 (50-69y)","C50"),
    ]:
        s=train[(train["age"]>=age_min)&(train["age"]<=age_max)].copy()
        s["age_group"]=pd.cut(s["age"],bins=[age_min-1,59,age_max],
                              labels=["50-59",f"60-{age_max}"]).astype(str)
        sub=s[ALL_FEAT+[target]].dropna()
        sub=sub.rename(columns={
            "age_group":"Age_group","region":"Region","zone":"Zone",
            "marital":"Marital_status","ethnicity":"Ethnicity",
            "education":"Education","insurance":"Health_insurance",
            "n_residents":"N_residents",target:"target"})
        path=OUT_PY_TAB/f"nomogram_train_{label}_ELSI.csv"
        sub.to_csv(path,index=False)
        print(f"  {label}: n={len(sub):,} pos={sub['target'].mean():.1%}")

        age_ref=f"{age_min}-59"
        r_script=f"""# Nomogram — {label} | Training: ELSI W1+W2 | {guideline}
if (!requireNamespace("rms",quietly=TRUE)) install.packages("rms")
if (!requireNamespace("Hmisc",quietly=TRUE)) install.packages("Hmisc")
library(rms); library(Hmisc)
options(contrasts=c("contr.treatment","contr.treatment"))

df <- read.csv("../../Outputs/Python_tables/nomogram_train_{label}_ELSI.csv")
cat_vars <- c("Zone","Marital_status","Health_insurance",
              "Age_group","Region","Ethnicity","Education")
df[cat_vars] <- lapply(df[cat_vars], as.factor)
df$Zone             <- relevel(df$Zone,             ref="Urban")
df$Marital_status   <- relevel(df$Marital_status,   ref="With partner")
df$Health_insurance <- relevel(df$Health_insurance, ref="Yes")
df$Region           <- relevel(df$Region,           ref="Southeast")
df$Ethnicity        <- relevel(df$Ethnicity,        ref="White")
df$Education        <- relevel(df$Education,        ref="Complete Elem")
df$Age_group        <- relevel(df$Age_group,        ref="{age_ref}")
dd <- datadist(df); options(datadist="dd")
f <- lrm(target ~ Zone + Marital_status + Health_insurance +
           Age_group + Region + Ethnicity + Education + N_residents,
         data=df, x=TRUE, y=TRUE)
print(f)
cat("\nC-statistic:", f$stats["C"], "\n")
print(round(exp(coef(f)[-1]),3))
png("../../Outputs/R_figures/{label}_nomogram_ELSI.png",width=2800,height=2000,res=220)
plot(nomogram(f,fun=plogis,funlabel="P(non-adherent)",N_residents=1:10),
     cex.axis=0.85,lmgp=0.25)
title("{label.replace("_"," ")} - Non-adherence nomogram\nELSI W1+W2 training data",cex.main=1.1)
dev.off()
png("../../Outputs/R_figures/{label}_calibration_ELSI.png",width=1400,height=1100,res=150)
plot(calibrate(f,method="boot",B=200),main="{label.replace("_"," ")} - Calibration")
dev.off()
cat("Done\n")
"""
        rpath=OUT_R/f"nomogram_{label}_ELSI.R"
        with open(rpath,"w") as fh: fh.write(r_script)
        print(f"  R script: {rpath.name}")

    print("  In RStudio (from Codes/R/):")
    print("    source('nomogram_Pap_smear_ELSI.R')")
    print("    source('nomogram_Mammography_ELSI.R')")


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    print("Loading datasets...")
    w1    = prep_elsi(ELSI_W1, 1)
    w2    = prep_elsi(ELSI_W2, 2)
    w3    = prep_elsi(ELSI_W3, 3)
    pns13 = prep_pns(PNS13, ";", 2013)
    pns19 = prep_pns(PNS19, ",", 2019)

    train = pd.concat([w1,w2], ignore_index=True)
    print(f"\nTraining pool (ELSI W1+W2): {len(train):,} women")

    all_res    = {}
    pipes_pap  = {}
    pipes_mamo = {}

    res_pap, pipes_pap = run_outcome(
        "pap_smear","target_pap",
        age_elsi=64,age_pns=64,
        train_data=train,w3=w3,pns13=pns13,pns19=pns19)
    all_res["pap_smear"] = res_pap

    res_mamo, pipes_mamo = run_outcome(
        "mammography","target_mamo",
        age_elsi=69,age_pns=69,
        train_data=train,w3=w3,pns13=pns13,pns19=pns19)
    all_res["mammography"] = res_mamo

    summary_table(all_res)
    barrier_analysis(pns13,pns19,pipes_pap,pipes_mamo)
    bootstrap_ci(w3,pns13,pns19,pipes_pap,pipes_mamo)
    export_risk_scores(pns19,pipes_pap,pipes_mamo)
    export_nomogram_data(train,pipes_pap,pipes_mamo)

    print(f"\nAll outputs saved to:")
    print(f"  Models:  {MODELS}")
    print(f"  Figures: {OUT_PY_FIG}")
    print(f"  Tables:  {OUT_PY_TAB}")
    print(f"  R codes: {OUT_R}")
