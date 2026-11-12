"""Contagens do fluxograma — mesmas regras do common.py. Saída: attrition_counts.csv
Etapas: raw, women, age50, <desfecho> disponível, n final (casos completos, idade-limite)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pyreadstat
from pathlib import Path
from common_legacy import *

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_TAB = ROOT / OUT_NAME / "Python_tables"; OUT_TAB.mkdir(parents=True, exist_ok=True)

def steps_for(raw, built, women_n, label):
    s = {"raw": len(raw), "women": women_n, "age50": len(built),
         "zone": int(built["zone"].notna().sum())}
    for key, tgt, amax in [("pap", "target_pap", PAP_MAX_AGE), ("mamo", "target_mamo", MAMO_MAX_AGE)]:
        s[f"{key}_avail"] = int(built[tgt].notna().sum())
        sub = built[(built["age"] <= amax) & built[tgt].notna()].copy()
        sub["age_group"] = pd.cut(sub["age"], bins=[AGE_MIN-1, AGE_CUT-1, amax],
                                  labels=["a", "b"]).astype(str)
        s[f"n_{key}"] = int(len(sub.dropna(subset=ALL_FEAT)))
    b = built.copy()
    b["age_group"] = pd.cut(b["age"], bins=[AGE_MIN-1, AGE_CUT-1, 200], labels=["a", "b"]).astype(str)
    both = b[b["target_pap"].notna() & b["target_mamo"].notna()]
    s["complete"] = int(len(both.dropna(subset=ALL_FEAT)))
    print(f"  {label}: Pap={s['n_pap']:,} | Mamo={s['n_mamo']:,}")
    return s

out = {}
for w, path in [(1, ELSI_W1), (2, ELSI_W2), (3, ELSI_W3)]:
    raw, _ = pyreadstat.read_dta(str(path), encoding="latin1")
    out[f"ELSI_W{w}"] = steps_for(raw, build_elsi(raw, w), int((num(raw["sexo"]) == 0).sum()), f"ELSI W{w}")
for y, path in [(2013, PNS13), (2019, PNS19)]:
    raw = load_pns_raw(path, y)
    out[f"PNS_{y}"] = steps_for(raw, build_pns(raw, y), int((num(raw["C006"]) == 2).sum()), f"PNS {y}")

rows = [{"dataset": d, "step": k, "n": v} for d, s in out.items() for k, v in s.items()]
pd.DataFrame(rows).to_csv(OUT_TAB/"attrition_counts.csv", index=False)
print("Saved attrition_counts.csv")
