"""R3-4: quantas mulheres responderam 'não sabe/ignorado' no desfecho e como diferem das incluídas.
Colocar em Codes/Python (usa common.py). Saída: Python_tables/dontknow_counts.csv (+ perfil)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pyreadstat
from pathlib import Path
import os
os.environ.setdefault("EDU_SCHEME", "4"); os.environ.setdefault("OUT_NAME", "Outputs_v3_edu4")
from common_legacy import *

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_TAB = ROOT / OUT_NAME / "Python_tables"; OUT_TAB.mkdir(parents=True, exist_ok=True)

def count(raw, built_idx, age, col_pap, col_mamo, label, pnsflag):
    rows = []
    for exam, col, amax in [("Pap smear", col_pap, PAP_MAX_AGE), ("Mammography", col_mamo, MAMO_MAX_AGE)]:
        v = num(raw[col]).loc[built_idx]
        a = age.loc[built_idx]
        sel = (a <= amax)
        v = v[sel]
        if pnsflag:
            v = v.dropna()   # PNS: NaN = não selecionada para o módulo individual
        rows.append(dict(dataset=label, exam=exam, women_asked=int(len(v)),
                         dont_know=int((v == 9).sum()), missing_or_na=int(v.isna().sum()) + int((v == 8).sum()),
                         pct_dont_know=round(100 * (v == 9).sum() / max(len(v), 1), 2)))
    return rows

out = []
for w, path in [(1, ELSI_W1), (2, ELSI_W2), (3, ELSI_W3)]:
    raw, _ = pyreadstat.read_dta(str(path), encoding="latin1")
    b = build_elsi(raw, w)
    out += count(raw, b.index, b["age"], "m13", "m15", f"ELSI W{w}", False)
for y, path in [(2013, PNS13), (2019, PNS19)]:
    raw = load_pns_raw(path, y)
    b = build_pns(raw, y)
    pc, mc = ("R001", "R017") if y == 2013 else ("R00101", "R01701")
    out += count(raw, b.index, b["age"], pc, mc, f"PNS {y}", True)
    # mamografia PNS: 'ignorado' também em R015 (fez mamografia?)
    v = num(raw["R015"]).loc[b.index].dropna()
    out.append(dict(dataset=f"PNS {y}", exam="Mammography (R015 ever done)", women_asked=int(len(v)),
                    dont_know=int((v == 9).sum()), missing_or_na=0,
                    pct_dont_know=round(100 * (v == 9).sum() / len(v), 2)))
res = pd.DataFrame(out); res.to_csv(OUT_TAB / "dontknow_counts.csv", index=False); print(res.to_string())

# Perfil: 'não sabe' x incluídas (ELSI W1+W2, Pap), idade/educação/plano
prof = []
for w, path in [(1, ELSI_W1), (2, ELSI_W2)]:
    raw, _ = pyreadstat.read_dta(str(path), encoding="latin1")
    b = build_elsi(raw, w); b["dk"] = (num(raw["m13"]).loc[b.index] == 9)
    b["incl"] = b["target_pap"].notna()
    prof.append(b[b["age"] <= PAP_MAX_AGE])
p = pd.concat(prof, ignore_index=True)
g = p[p.dk | p.incl].assign(grupo=lambda d: np.where(d.dk, "dont_know", "included"))
print(g.groupby("grupo")["age"].agg(["count","mean"]).round(2))
for c in ["education", "insurance", "region"]:
    print(pd.crosstab(g[c], g["grupo"], normalize="columns").round(3))
