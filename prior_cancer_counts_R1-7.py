"""R1-7: mulheres com câncer prévio nas amostras PNS (ELSI: aguardando dicionário).
PNS 2013: Q120 (algum diagnóstico de câncer; 1=Sim), Q121 (1º tipo: 4=Mama, 5=Colo). 
PNS 2019: Q120, Q12107 (mama), Q12108 (colo), Q121014 (ovário). 9=Ignorado.
Colocar em Codes/Python. Saída: Python_tables/prior_cancer_counts.csv"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
import os
os.environ.setdefault("EDU_SCHEME", "4"); os.environ.setdefault("OUT_NAME", "Outputs_v3_edu4")
from common_legacy import *

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_TAB = ROOT / OUT_NAME / "Python_tables"; OUT_TAB.mkdir(parents=True, exist_ok=True)

EXTRA = {2013: ["Q120", "Q121"], 2019: ["Q120", "Q12107", "Q12108", "Q121014"]}
rows = []
for y, path in [(2013, PNS13), (2019, PNS19)]:
    cols = PNS_COLS | set(EXTRA[y])
    raw = pd.read_csv(path, sep=";" if y == 2013 else ",", encoding="latin1",
                      usecols=lambda c: c in cols, dtype=str)
    b = build_pns(raw, y)
    q = {c: num(raw[c]).loc[b.index] for c in EXTRA[y]}
    any_c = q["Q120"] == 1
    if y == 2013:
        breast, cervix = any_c & (q["Q121"] == 4), any_c & (q["Q121"] == 5)
    else:
        breast, cervix = q["Q12107"] == 1, q["Q12108"] == 1
    for exam, tgt, amax, site in [("Pap smear", "target_pap", PAP_MAX_AGE, cervix),
                                  ("Mammography", "target_mamo", MAMO_MAX_AGE, breast)]:
        m = b[tgt].notna() & (b["age"] <= amax)
        m &= b[[c for c in ALL_FEAT if c != "age_group"]].notna().all(axis=1)   # age_group é derivado da idade
        n = int(m.sum())
        for lab, flag in [("any cancer (Q120)", any_c), ("site-specific (cervix/breast)", site)]:
            f = flag & m
            rows.append(dict(dataset=f"PNS {y}", exam=exam, n_analytic=n, group=lab,
                             n_prior_cancer=int(f.sum()), pct=round(100 * f.sum() / n, 2),
                             nonadh_pct_among_prior=round(100 * b.loc[f, tgt].mean(), 1) if f.sum() else np.nan,
                             nonadh_pct_others=round(100 * b.loc[m & ~flag, tgt].mean(), 1)))
        rows.append(dict(dataset=f"PNS {y}", exam=exam, n_analytic=n, group="Q120 ignorado (9)",
                         n_prior_cancer=int(((q["Q120"] == 9) & m).sum()),
                         pct=round(100 * ((q["Q120"] == 9) & m).sum() / n, 2)))
res = pd.DataFrame(rows); res.to_csv(OUT_TAB / "prior_cancer_counts.csv", index=False)
print(res.to_string())

# ---------------------------------------------------------------- ELSI
# n60: "Algum médico já disse que o(a) Sr(a) tem ou teve câncer?" 0 Não / 1 Sim / 9 Não sabe (W1, W2, W3)
# W3 apenas: n60_1_1 mama, n60_1_2 útero (colo ou corpo), n60_1_3 ovário (mulheres)
import pyreadstat
rows = []
for w, path in [(1, ELSI_W1), (2, ELSI_W2), (3, ELSI_W3)]:
    raw, _ = pyreadstat.read_dta(str(path), encoding="latin1")
    b = build_elsi(raw, w)
    n60 = num(raw["n60"]).loc[b.index]
    site = {}
    if w == 3:
        site = {"breast (n60_1_1)": num(raw["n60_1_1"]).loc[b.index] == 1,
                "uterus (n60_1_2)": num(raw["n60_1_2"]).loc[b.index] == 1}
    for exam, tgt, amax in [("Pap smear", "target_pap", PAP_MAX_AGE), ("Mammography", "target_mamo", MAMO_MAX_AGE)]:
        m = b[tgt].notna() & (b["age"] <= amax)
        m &= b[[c for c in ALL_FEAT if c != "age_group"]].notna().all(axis=1)
        n = int(m.sum())
        groups = [("any cancer (n60=1)", n60 == 1), ("n60 not known (9)", n60 == 9)] + list(site.items())
        for lab, flag in groups:
            f = flag & m
            rows.append(dict(dataset=f"ELSI W{w}", exam=exam, n_analytic=n, group=lab, n_prior_cancer=int(f.sum()),
                             pct=round(100 * f.sum() / n, 2),
                             nonadh_pct_among_prior=round(100 * b.loc[f, tgt].mean(), 1) if f.sum() else np.nan,
                             nonadh_pct_others=round(100 * b.loc[m & ~flag, tgt].mean(), 1)))
res2 = pd.DataFrame(rows); print(res2.to_string())
pd.concat([res, res2]).to_csv(OUT_TAB / "prior_cancer_counts.csv", index=False)
