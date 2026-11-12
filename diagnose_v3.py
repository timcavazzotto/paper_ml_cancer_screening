"""
diagnose.py — roda ANTES do retreino. Imprime, por amostra, o que o 'common.py'
fez com cada código bruto. Envie a saída completa para conferência.
Uso: python diagnose.py --elsi1 W1.dta --elsi2 W2.dta --elsi3 W3.dta \
                        --pns2013 pns2013.csv --pns2019 pns2019.csv
"""
import argparse, sys
import pandas as pd, numpy as np, pyreadstat
from common import *

ap = argparse.ArgumentParser()
add_path_args(ap)
a = ap.parse_args()

def vc(title, s):
    print(f"\n--- {title}"); print(s.value_counts(dropna=False).sort_index().to_string())

raw = {}
for w, p in [(1, a.elsi1), (2, a.elsi2), (3, a.elsi3)]:
    df, meta = pyreadstat.read_dta(p, encoding="latin1"); raw[f"E{w}"] = df
    print(f"\n==================== ELSI W{w}  (n bruto={len(df)})")
    for v in ["m13", "m15", "u1", "e22", "e7", "e9", "ar6" if w == 1 else "nmoradores"]:
        if v in df:
            vc(v, df[v])
            lab = meta.variable_value_labels.get(v)
            if lab: print("   rótulos:", lab)
        else:
            print(f"!! variável {v} AUSENTE")
    d = build_elsi(df, w)
    print("\n  Crosstab m15 bruto x target_mamo"); print(pd.crosstab(num(df.loc[d.index, "m15"]), d["target_mamo"], dropna=False))
    print("  Crosstab m13 bruto x target_pap");  print(pd.crosstab(num(df.loc[d.index, "m13"]), d["target_pap"], dropna=False))
    vc("education derivada", d["education"]); vc("insurance derivada", d["insurance"])
    vc("ethnicity derivada", d["ethnicity"]); vc("marital derivada", d["marital"])
    print("  peso: NaN =", d["weight"].isna().sum(), "| média =", round(d["weight"].mean(), 3))

for y, p in [(2013, a.pns2013), (2019, a.pns2019)]:
    df = load_pns_raw(p, y)
    print(f"\n==================== PNS {y} (n bruto={len(df)})")
    cols = (["C009", "C010", "I001", "I002", "I003", "R001", "R010", "R015", "R017", "R002", "VDD004A", "V0022"]
            if y == 2013 else
            ["C009", "C01001", "I00102", "R00101", "R010", "R015", "R01701", "R002010", "VDD004A", "V0022"])
    for v in cols:
        if v in df: vc(v, pd.to_numeric(df[v].str.strip().str.replace(',', '.'), errors='coerce'))
        else: print(f"!! variável {v} AUSENTE")
    d = build_pns(df, y)
    mc = "R017" if y == 2013 else "R01701"
    print("\n  Crosstab R015 x mamografia-tempo x target_mamo")
    print(pd.crosstab([num(df.loc[d.index, "R015"]), num(df.loc[d.index, mc])], d["target_mamo"], dropna=False))
    pc = "R001" if y == 2013 else "R00101"
    print("  Crosstab Pap bruto x R010 x target_pap")
    print(pd.crosstab([num(df.loc[d.index, pc]), num(df.loc[d.index, "R010"])], d["target_pap"], dropna=False))
    vc("education derivada", d["education"]); vc("insurance derivada", d["insurance"])
    vc("marital derivada", d["marital"]); vc("ethnicity derivada", d["ethnicity"])
    print("  peso: NaN =", d["weight"].isna().sum(), "| mulheres >=50:", len(d))
    for t in ["target_pap", "target_mamo"]:
        ok = d[t].notna()
        print(f"  {t}: desfecho válido={ok.sum()} | sem peso entre eles={(ok & d['weight'].isna()).sum()}")
