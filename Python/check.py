import pandas as pd
import numpy as np
import pyreadstat


import pyreadstat

base = r"C:\Users\Usuario\Documents\ARQUIVOS TIMOTHY\DOCÊNCIA\UNICENTRO\2026\DEMED\Pesquisa\Timothy\Pamella\Final_5_samples\Data"

for wave, fname in [
    (2, "ELSI Portugues (2a onda) stata13.dta"),
    (3, "ELSI Portugues (3a onda).dta"),
]:
    df, _ = pyreadstat.read_dta(f"{base}\\{fname}", encoding="latin1")
    print(f"\nW{wave} u1:")
    print(df["u1"].value_counts(dropna=False).sort_index())




