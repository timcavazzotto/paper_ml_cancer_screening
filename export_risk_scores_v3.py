"""Risco médio previsto por UF (PNS 2019) — usa common.py. Saída: risk_scores_uf.csv"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
from pathlib import Path
from common_legacy import *

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_TAB = ROOT / OUT_NAME / "Python_tables"; MODELS = ROOT / OUT_NAME / "Models"
UF_SIGLA = {11:"RO",12:"AC",13:"AM",14:"RR",15:"PA",16:"AP",17:"TO",21:"MA",22:"PI",23:"CE",
            24:"RN",25:"PB",26:"PE",27:"AL",28:"SE",29:"BA",31:"MG",32:"ES",33:"RJ",35:"SP",
            41:"PR",42:"SC",43:"RS",50:"MS",51:"MT",52:"GO",53:"DF"}
pns19 = read_pns(PNS19, 2019)
pipes = {"pap":  joblib.load(MODELS/"model_pap_smear_XGBoost.joblib"),
         "mamo": joblib.load(MODELS/"model_mammography_XGBoost.joblib")}
rows = []
for uf, grp in pns19.groupby("uf"):
    row = {"uf": int(uf), "sigla": UF_SIGLA.get(int(uf), "??"),
           "regiao": grp["region"].dropna().mode()[0] if grp["region"].notna().any() else None}
    for key, age_lim in [("pap", 64), ("mamo", 69)]:
        g = grp[grp["age"] <= age_lim].copy()
        g["age_group"] = pd.cut(g["age"], bins=[AGE_MIN-1, AGE_CUT-1, age_lim],
                                labels=[f"{AGE_MIN}-{AGE_CUT-1}", f"{AGE_CUT}-{age_lim}"]).astype(str)
        g = g.dropna(subset=ALL_FEAT)
        row[f"risk_{key}"] = pipes[key].predict_proba(g[ALL_FEAT])[:, 1].mean() if len(g) >= 30 else np.nan
        row[f"n_{key}"] = len(g)
    rows.append(row)
out = pd.DataFrame(rows); out.to_csv(OUT_TAB/"risk_scores_uf.csv", index=False)
print(f"Saved risk_scores_uf.csv ({len(out)} UFs)")
print(out.to_string(index=False))
