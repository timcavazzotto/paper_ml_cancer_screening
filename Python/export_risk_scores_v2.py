"""
Export risk scores by state (UF) for ecological analysis
New design: models trained on ELSI W1+W2, applied to PNS 2019
Output: Outputs/Python_tables/risk_scores_uf.csv
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

ROOT      = Path(__file__).parent.parent.parent
DATA      = ROOT / "Data"
OUT_TAB   = ROOT / "Outputs" / "Python_tables"
MODELS    = ROOT / "Outputs" / "Models"
PNS19     = DATA / "pns2019.csv"

MISS = {9,99,999,88,8}
def clean(s): return pd.to_numeric(s, errors="coerce").replace(list(MISS), np.nan)

ETHNIC_PNS  = {1:"White",2:"Black",3:"Yellow",4:"Brown",5:"Indigenous"}
MAR_PNS19   = {1:"With partner",2:"Without partner",
               3:"Without partner",4:"Without partner"}
INS_PNS     = {1:"Yes",2:"No"}
ZONE_R      = {1:"Urban",2:"Rural"}
EDUC_PNS    = {1:"Illiterate",2:"Incomplete Elem",3:"Complete Elem",
               4:"Incomplete Elem",5:"Complete Elem",
               6:"Higher Educ",7:"Higher Educ"}
UF_REG      = {11:"North",12:"North",13:"North",14:"North",15:"North",
               16:"North",17:"North",
               21:"Northeast",22:"Northeast",23:"Northeast",24:"Northeast",
               25:"Northeast",26:"Northeast",27:"Northeast",28:"Northeast",
               29:"Northeast",31:"Southeast",32:"Southeast",33:"Southeast",
               35:"Southeast",41:"South",42:"South",43:"South",
               50:"Central-West",51:"Central-West",52:"Central-West",
               53:"Central-West"}

CAT_FEAT = ["age_group","region","zone","marital","ethnicity",
            "education","insurance"]
NUM_FEAT = ["n_residents"]
ALL_FEAT = CAT_FEAT + NUM_FEAT

print("Loading PNS 2019...")
df = pd.read_csv(PNS19, sep=",", encoding="latin1", low_memory=False)

o = pd.DataFrame()
o["age"]       = clean(df["C008"])
o["sex"]       = clean(df["C006"])
o["ethnicity"] = clean(df["C009"]).map(ETHNIC_PNS)
o["marital"]   = clean(df["C011"]).map(MAR_PNS19)
o["zone"]      = clean(df["V0026"]).map(ZONE_R)
o["region"]    = pd.to_numeric(df["V0001"],errors="coerce").map(UF_REG)
o["education"] = clean(df["VDD004A"]).map(EDUC_PNS)
o["insurance"] = clean(df["I00102"]).map(INS_PNS)
o["n_residents"]= clean(df["V0022"]).clip(upper=10)
o["uf"]        = pd.to_numeric(df["V0001"],errors="coerce").astype("Int64")

pns19 = o[(o["sex"]==2)&(o["age"]>=50)].copy()
print(f"  Women >= 50: {len(pns19):,}")

print("Loading models...")
pipe_pap  = joblib.load(MODELS/"model_pap_smear_XGBoost.joblib")
pipe_mamo = joblib.load(MODELS/"model_mammography_XGBoost.joblib")

UF_SIGLA = {
    11:"RO",12:"AC",13:"AM",14:"RR",15:"PA",16:"AP",17:"TO",
    21:"MA",22:"PI",23:"CE",24:"RN",25:"PB",26:"PE",27:"AL",
    28:"SE",29:"BA",31:"MG",32:"ES",33:"RJ",35:"SP",
    41:"PR",42:"SC",43:"RS",50:"MS",51:"MT",52:"GO",53:"DF"
}

print("Computing risk scores by UF...")
results=[]
for uf_code, grp in pns19.groupby("uf"):
    row={"uf":int(uf_code),"sigla":UF_SIGLA.get(int(uf_code),"??"),
         "regiao":grp["region"].dropna().mode()[0]
         if grp["region"].notna().any() else None}

    for outcome,pipe,age_lim,col in [
        ("pap", pipe_pap,  64,"risk_pap"),
        ("mamo",pipe_mamo, 69,"risk_mamo"),
    ]:
        g=grp[grp["age"]<=age_lim].copy()
        g["age_group"]=pd.cut(g["age"],bins=[49,59,age_lim],
                              labels=["50-59",f"60-{age_lim}"]).astype(str)
        g=g.dropna(subset=ALL_FEAT)
        if len(g)>=30:
            row[col]=pipe.predict_proba(g[ALL_FEAT])[:,1].mean()
            row[f"n_{outcome}"]=len(g)
        else:
            row[col]=np.nan; row[f"n_{outcome}"]=len(g)
    results.append(row)

df_out=pd.DataFrame(results)
df_out.to_csv(OUT_TAB/"risk_scores_uf.csv",index=False)
print(f"Saved: risk_scores_uf.csv ({len(df_out)} UFs)")
print(df_out[["uf","sigla","regiao","risk_pap","risk_mamo","n_pap","n_mamo"]].to_string(index=False))
