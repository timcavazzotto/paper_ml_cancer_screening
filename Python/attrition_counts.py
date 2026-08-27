"""
Attrition counts — same filters as pns_pipeline_v2.py
Outputs: Outputs/Python_tables/attrition_counts.csv
         (used by master_analysis.R to draw the flowchart)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pyreadstat
from pathlib import Path

ROOT    = Path(__file__).parent.parent.parent
DATA    = ROOT / "Data"
OUT_TAB = ROOT / "Outputs" / "Python_tables"
OUT_TAB.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# SAME CONSTANTS AS pns_pipeline_v2.py
# ==============================================================================
MISS = {9,99,999,88,8}
def clean(s): return pd.to_numeric(s, errors="coerce").replace(list(MISS), np.nan)

ETHNIC_ELSI = {1:"White",2:"Black",3:"Brown",4:"Yellow",5:"Indigenous"}
ETHNIC_PNS  = {1:"White",2:"Black",3:"Yellow",4:"Brown", 5:"Indigenous"}
MAR_ELSI    = {1:"Without partner",2:"With partner",
               3:"Without partner",4:"Without partner"}
MAR_PNS13   = {1:"With partner",2:"Without partner",3:"Without partner",
               4:"Without partner",5:"Without partner"}
MAR_PNS19   = {1:"With partner",2:"Without partner",
               3:"Without partner",4:"Without partner"}
INS_ELSI    = {0:"No",1:"Yes"}
INS_PNS     = {1:"Yes",2:"No"}
ZONE_R      = {1:"Urban",2:"Rural"}
REG_ELSI    = {1:"North",2:"Northeast",3:"Southeast",4:"South",5:"Central-West"}
UF_REG      = {11:"North",12:"North",13:"North",14:"North",15:"North",
               16:"North",17:"North",
               21:"Northeast",22:"Northeast",23:"Northeast",24:"Northeast",
               25:"Northeast",26:"Northeast",27:"Northeast",28:"Northeast",
               29:"Northeast",31:"Southeast",32:"Southeast",33:"Southeast",
               35:"Southeast",41:"South",42:"South",43:"South",
               50:"Central-West",51:"Central-West",52:"Central-West",
               53:"Central-West"}
EDUC_PNS    = {1:"Illiterate",2:"Incomplete Elem",3:"Complete Elem",
               4:"Incomplete Elem",5:"Complete Elem",
               6:"Higher Educ",7:"Higher Educ"}

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
# ATTRITION — ELSI
# ==============================================================================
def attrition_elsi(path, wave):
    print(f"  ELSI W{wave}...")
    df, _ = pyreadstat.read_dta(str(path), encoding="latin1")
    steps = {}

    steps["raw"] = len(df)

    # Women >= 50
    df = df[pd.to_numeric(df["sexo"], errors="coerce") == 0].copy()
    steps["women"] = len(df)

    df = df[clean(df["idade"]) >= 50].copy()
    steps["age50"] = len(df)

    # Pap outcome (m13)
    m13 = clean(df["m13"])
    df["target_pap"] = m13.isin({2,3}).astype("Int64")
    df["target_pap"] = df["target_pap"].where(m13.isin({1,2,3}), np.nan)
    df_pap = df[df["target_pap"].notna()].copy()
    steps["pap_avail"] = len(df_pap)

    # Mamo outcome (m15)
    # Mamo — counted independently from age50 (not requiring pap)
    m15_all = clean(df["m15"])  # df here = age50 filtered
    df_mamo_src = df.copy()
    if wave in [1, 2]:
        df_mamo_src["target_mamo"] = m15_all.isin({1,5}).astype("Int64")
    else:
        df_mamo_src["target_mamo"] = m15_all.isin({4,5}).astype("Int64")
    df_mamo_src["target_mamo"] = df_mamo_src["target_mamo"].where(
        m15_all.isin({1,2,3,4,5}), np.nan)
    df_mamo_src = df_mamo_src[df_mamo_src["target_mamo"].notna()].copy()
    steps["mamo_avail"] = len(df_mamo_src)

    # Also keep df_both for complete step display
    m15 = clean(df_pap["m15"])
    if wave in [1, 2]:
        df_pap["target_mamo"] = m15.isin({1,5}).astype("Int64")
    else:
        df_pap["target_mamo"] = m15.isin({4,5}).astype("Int64")
    df_pap["target_mamo"] = df_pap["target_mamo"].where(
        m15.isin({1,2,3,4,5}), np.nan)
    df_both = df_pap[df_pap["target_mamo"].notna()].copy()

    # Build features
    df_both["ethnicity"] = clean(df_both["e9"]).map(ETHNIC_ELSI)
    df_both["marital"]   = clean(df_both["e7"]).map(MAR_ELSI)
    df_both["zone"]      = clean(df_both["zona"]).map(ZONE_R)
    df_both["region"]    = clean(df_both["regiao"]).map(REG_ELSI)
    df_both["education"] = clean(df_both["e22"]).apply(educ_elsi)
    df_both["insurance"] = clean(df_both["u1"]).map(INS_ELSI)
    res = "ar6" if wave==1 else "nmoradores"
    df_both["n_residents"] = clean(df_both[res]).clip(upper=10)

    # Complete predictors (on combined dataset)
    feat_cols = ["ethnicity","marital","zone","region",
                 "education","insurance","n_residents"]

    # Count pap and mamo INDEPENDENTLY — matching get_Xy logic
    # Pap: from pap_avail, complete predictors, age <= 64
    df_pap_feat = df_pap.copy()
    df_pap_feat["ethnicity"] = clean(df_pap_feat["e9"]).map(ETHNIC_ELSI)
    df_pap_feat["marital"]   = clean(df_pap_feat["e7"]).map(MAR_ELSI)
    df_pap_feat["zone"]      = clean(df_pap_feat["zona"]).map(ZONE_R)
    df_pap_feat["region"]    = clean(df_pap_feat["regiao"]).map(REG_ELSI)
    df_pap_feat["education"] = clean(df_pap_feat["e22"]).apply(educ_elsi)
    df_pap_feat["insurance"] = clean(df_pap_feat["u1"]).map(INS_ELSI)
    res = "ar6" if wave==1 else "nmoradores"
    df_pap_feat["n_residents"] = clean(df_pap_feat[res]).clip(upper=10)
    age_pap = clean(df_pap_feat["idade"])
    df_pap_feat["age_group"] = pd.cut(age_pap, bins=[49,59,64],
                                       labels=["50-59","60-64"]).astype(str)
    sub_pap = df_pap_feat[
        age_pap<=64].dropna(subset=ALL_FEAT+["target_pap"])
    steps["n_pap"] = len(sub_pap)

    # Mamo: from mamo_avail (independent), complete predictors, age <= 69
    df_mamo_feat = df_mamo_src.copy()
    df_mamo_feat["ethnicity"] = clean(df_mamo_feat["e9"]).map(ETHNIC_ELSI)
    df_mamo_feat["marital"]   = clean(df_mamo_feat["e7"]).map(MAR_ELSI)
    df_mamo_feat["zone"]      = clean(df_mamo_feat["zona"]).map(ZONE_R)
    df_mamo_feat["region"]    = clean(df_mamo_feat["regiao"]).map(REG_ELSI)
    df_mamo_feat["education"] = clean(df_mamo_feat["e22"]).apply(educ_elsi)
    df_mamo_feat["insurance"] = clean(df_mamo_feat["u1"]).map(INS_ELSI)
    res = "ar6" if wave==1 else "nmoradores"
    df_mamo_feat["n_residents"] = clean(df_mamo_feat[res]).clip(upper=10)
    age_mamo = clean(df_mamo_feat["idade"])
    df_mamo_feat["age_group"] = pd.cut(age_mamo, bins=[49,59,69],
                                        labels=["50-59","60-69"]).astype(str)
    sub_mamo = df_mamo_feat[
        age_mamo<=69].dropna(subset=ALL_FEAT+["target_mamo"])
    steps["n_mamo"] = len(sub_mamo)

    # Complete = union for display purposes
    df_complete = df_both.dropna(subset=feat_cols).copy()
    steps["complete"] = len(df_complete)

    print(f"    Pap={steps['n_pap']:,} | Mamo={steps['n_mamo']:,}")
    return steps

# ==============================================================================
# ATTRITION — PNS
# ==============================================================================
def attrition_pns(path, sep, year):
    print(f"  PNS {year}...")
    df = pd.read_csv(path, sep=sep, encoding="latin1", low_memory=False)
    steps = {}

    steps["raw"] = len(df)

    # Women
    df = df[clean(df["C006"]) == 2].copy()
    steps["women"] = len(df)

    # Age >= 50
    df = df[clean(df["C008"]) >= 50].copy()
    steps["age50"] = len(df)

    # Zone available
    df = df[clean(df["V0026"]).isin([1,2])].copy()
    steps["zone"] = len(df)

    # Pap outcome + hysterectomy exclusion
    if year == 2013:
        pap = clean(df["R001"])
        df["target_pap"] = pap.isin([4,5]).astype("Int64")
        df["target_pap"] = df["target_pap"].where(pap.isin([1,2,3,4,5]), np.nan)
        r002 = pd.to_numeric(df["R002"], errors="coerce")
        df["target_pap"] = df["target_pap"].where(r002 != 12, np.nan)
    else:
        pap = clean(df["R00101"])
        df["target_pap"] = pap.isin([4,5]).astype("Int64")
        df["target_pap"] = df["target_pap"].where(pap.isin([1,2,3,4,5]), np.nan)
        r002010 = pd.to_numeric(df["R002010"], errors="coerce")
        df["target_pap"] = df["target_pap"].where(r002010 != 12, np.nan)

    df_pap = df[df["target_pap"].notna()].copy()
    steps["pap_avail"] = len(df_pap)

    # Mamo outcome
    md = clean(df_pap["R015"])
    mw_col = "R017" if year==2013 else "R01701"
    mw = clean(df_pap[mw_col])
    df_pap["target_mamo"] = ((md==2)|mw.isin([3,4])).astype("Int64")
    df_pap["target_mamo"] = df_pap["target_mamo"].where(
        md.isin([1,2])|mw.isin([1,2,3,4]), np.nan)
    df_both = df_pap[df_pap["target_mamo"].notna()].copy()
    steps["mamo_avail"] = len(df_both)

    # Build features
    df_both["ethnicity"]  = clean(df_both["C009"]).map(ETHNIC_PNS)
    df_both["zone"]       = clean(df_both["V0026"]).map(ZONE_R)
    df_both["region"]     = pd.to_numeric(df_both["V0001"],
                             errors="coerce").map(UF_REG)
    df_both["education"]  = clean(df_both["VDD004A"]).map(EDUC_PNS)
    df_both["n_residents"]= clean(df_both["V0022"]).clip(upper=10)

    if year == 2013:
        df_both["marital"]  = clean(df_both["C011"]).map(MAR_PNS13)
        df_both["insurance"]= clean(df_both["I001"]).map(INS_PNS)
    else:
        df_both["marital"]  = clean(df_both["C011"]).map(MAR_PNS19)
        df_both["insurance"]= clean(df_both["I00102"]).map(INS_PNS)

    # Build features on both datasets
    feat_cols = ["ethnicity","marital","zone","region",
                 "education","insurance","n_residents"]

    def add_pns_features(df, year):
        df = df.copy()
        df["ethnicity"]  = clean(df["C009"]).map(ETHNIC_PNS)
        df["zone"]       = clean(df["V0026"]).map(ZONE_R)
        df["region"]     = pd.to_numeric(df["V0001"],
                           errors="coerce").map(UF_REG)
        df["education"]  = clean(df["VDD004A"]).map(EDUC_PNS)
        df["n_residents"]= clean(df["V0022"]).clip(upper=10)
        if year==2013:
            df["marital"]  = clean(df["C011"]).map(MAR_PNS13)
            df["insurance"]= clean(df["I001"]).map(INS_PNS)
        else:
            df["marital"]  = clean(df["C011"]).map(MAR_PNS19)
            df["insurance"]= clean(df["I00102"]).map(INS_PNS)
        return df

    df_pap  = add_pns_features(df_pap,  year)
    df_both = add_pns_features(df_both, year)

    # Pap: from pap_avail, complete predictors, age <= 64
    age_pap = clean(df_pap["C008"])
    df_pap["age_group"] = pd.cut(age_pap, bins=[49,59,64],
                                  labels=["50-59","60-64"]).astype(str)
    sub_pap = df_pap[age_pap<=64].dropna(
        subset=ALL_FEAT+["target_pap"])
    steps["n_pap"] = len(sub_pap)

    # Mamo: independent from pap — start from zone-filtered dataset
    df_mamo_src = df.copy()  # df = zone-filtered
    md_all = clean(df_mamo_src["R015"])
    mw_col = "R017" if year==2013 else "R01701"
    mw_all = clean(df_mamo_src[mw_col])
    df_mamo_src["target_mamo"] = ((md_all==2)|mw_all.isin([3,4])).astype("Int64")
    df_mamo_src["target_mamo"] = df_mamo_src["target_mamo"].where(
        md_all.isin([1,2])|mw_all.isin([1,2,3,4]), np.nan)
    df_mamo_src = df_mamo_src[df_mamo_src["target_mamo"].notna()].copy()
    df_mamo_src = add_pns_features(df_mamo_src, year)
    age_mamo = clean(df_mamo_src["C008"])
    df_mamo_src["age_group"] = pd.cut(age_mamo, bins=[49,59,69],
                                       labels=["50-59","60-69"]).astype(str)
    sub_mamo = df_mamo_src[age_mamo<=69].dropna(
        subset=ALL_FEAT+["target_mamo"])
    steps["n_mamo"] = len(sub_mamo)

    # Complete = mamo_avail with complete predictors for display
    df_complete = df_both.dropna(subset=feat_cols).copy()
    steps["complete"] = len(df_complete)

    print(f"    Pap={steps['n_pap']:,} | Mamo={steps['n_mamo']:,}")
    return steps

# ==============================================================================
# RUN
# ==============================================================================
print("Computing attrition counts...\n")

datasets = {
    "ELSI_W1": attrition_elsi(
        DATA/"ELSI Portugues (1a onda) stata13.dta", 1),
    "ELSI_W2": attrition_elsi(
        DATA/"ELSI Portugues (2a onda) stata13.dta", 2),
    "ELSI_W3": attrition_elsi(
        DATA/"ELSI Portugues (3a onda).dta", 3),
    "PNS_2013": attrition_pns(
        DATA/"PNS_2013.csv", ";", 2013),
    "PNS_2019": attrition_pns(
        DATA/"pns2019.csv", ",", 2019),
}

# Save as CSV for R
rows = []
for ds, steps in datasets.items():
    for step, val in steps.items():
        rows.append({"dataset":ds, "step":step, "n":val})

df_out = pd.DataFrame(rows)
out_path = OUT_TAB / "attrition_counts.csv"
df_out.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

# Print summary
print("\n=== SUMMARY ===")
for ds, steps in datasets.items():
    print(f"\n{ds}:")
    for k,v in steps.items():
        print(f"  {k:15s}: {v:,}")
