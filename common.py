"""
common.py — regras de codificação ÚNICAS para todos os scripts (v3)
====================================================================
Todos os scripts (treino, Tabela 1, ROC, fluxograma, risco por UF, pesos)
devem fazer `from common import *` e NÃO reescrever nenhuma regra abaixo.

Decisões incorporadas (20/10/2026):
  A1  clean() só apaga códigos de não resposta POR VARIÁVEL (não mais um
      conjunto global {9,99,999,88,8}).
  A2  Mamografia: corte bienal (2 anos) em TODAS as amostras.
  A3  Histerectomia: R010 != 1 (PNS 2013 e 2019), só no Pap.
  A4  Escolaridade em 3 categorias (opção A).
  A5  Plano de saúde PNS 2013: exclui quem só tem plano odontológico.
  A6  ELSI u1: codificação detectada por onda (0/1 ou 1/2), com erro se ambígua.
  A7  Raça: Amarela + Indígena -> "Other".
  B3  Parceiro: PNS C010 (2013) / C01001 (2019); ELSI e7 == 2.
"""
import os
import numpy as np
import pandas as pd

# Variáveis de ambiente (análises de sensibilidade; padrão = análise principal)
EDU_SCHEME = os.environ.get("EDU_SCHEME", "3")           # "3" ou "4" categorias de escolaridade
if EDU_SCHEME not in ("3", "4"):
    raise SystemExit("EDU_SCHEME deve ser '3' ou '4'")
# Pasta de saída: o padrão acompanha o esquema, para nunca misturar resultados
OUT_NAME   = os.environ.get("OUT_NAME", "Outputs_v3" if EDU_SCHEME == "3" else "Outputs_v3_edu4")
if EDU_SCHEME == "4" and OUT_NAME == "Outputs_v3":
    raise SystemExit("PARADO: EDU_SCHEME=4 gravaria em Outputs_v3 (resultados de 3 categorias). "
                     "Defina OUT_NAME=Outputs_v3_edu4.")
if EDU_SCHEME == "3" and OUT_NAME.endswith("edu4"):
    raise SystemExit("PARADO: EDU_SCHEME=3 com pasta *_edu4. Confira EDU_SCHEME e OUT_NAME.")
print(f"[common] escolaridade = {EDU_SCHEME} categorias | saída = {OUT_NAME}")

# ------------------------------------------------------------------ CONFIG
AGE_CUT      = 60      # corte do grupo etário (R3-5 ainda em aberto)
AGE_MIN      = 50
PAP_MAX_AGE  = 64
MAMO_MAX_AGE = 69
ELSI_WEIGHT  = "peso_calibrado"
PNS_WEIGHT   = "V00291"

CAT_FEAT = ["age_group", "region", "zone", "marital", "ethnicity",
            "education", "insurance"]
NUM_FEAT = ["n_residents"]
ALL_FEAT = CAT_FEAT + NUM_FEAT

# Mamografia — categorias de NÃO ADERENTE (corte de 2 anos = bienal MS/INCA)
#   ELSI W1/W2 m15: 1 Nunca, 2 <1a, 3 1-2a, 4 2-<3a, 5 >=3a
#   ELSI W3    m15: 1 <1a,  2 1-2a, 3 2-<3a, 4 >=3a, 5 Nunca
#   PNS R017/R01701: 1 <1a, 2 1-2a, 3 2-<3a, 4 >=3a  (R015==2 -> nunca fez)
MAMO_NONADH_ELSI = {1: {1, 4, 5}, 2: {1, 4, 5}, 3: {3, 4, 5}}
MAMO_VALID_ELSI  = {1, 2, 3, 4, 5}
MAMO_NONADH_PNS  = {3, 4}
# Pap — corte de 3 anos (INCA 2016)
#   ELSI m13: 1 <=3a, 2 >3a, 3 Nunca    PNS R001/R00101: 1,2,3 <3a; 4 >=3a; 5 Nunca
PAP_NONADH_ELSI  = {2, 3}
PAP_VALID_ELSI   = {1, 2, 3}
PAP_NONADH_PNS   = {4, 5}
PAP_VALID_PNS    = {1, 2, 3, 4, 5}

# Categorias
EDU_LOW  = "Up to incomplete elementary"
EDU_MID  = "Complete elementary to secondary"
EDU_HIGH = "Higher education"
EDU_NONE = "No schooling"
EDU_INC  = "Incomplete elementary"
if EDU_SCHEME == "4":      # sensibilidade: 4 categorias monotônicas
    EDUC_PNS = {1: EDU_NONE, 2: EDU_INC, 3: EDU_MID, 4: EDU_MID, 5: EDU_MID,
                6: EDU_HIGH, 7: EDU_HIGH}
else:                      # principal: 3 categorias
    EDUC_PNS = {1: EDU_LOW, 2: EDU_LOW, 3: EDU_MID, 4: EDU_MID, 5: EDU_MID,
                6: EDU_HIGH, 7: EDU_HIGH}
ETHNIC_ELSI = {1: "White", 2: "Black", 3: "Brown", 4: "Other", 5: "Other"}
ETHNIC_PNS  = {1: "White", 2: "Black", 3: "Other", 4: "Brown", 5: "Other"}
MAR_ELSI    = {1: "Without partner", 2: "With partner",
               3: "Without partner", 4: "Without partner"}
MAR_PNS     = {1: "With partner", 2: "Without partner"}   # C010 / C01001
INS_PNS     = {1: "Yes", 2: "No"}
ZONE_R      = {1: "Urban", 2: "Rural"}
REG_ELSI    = {1: "North", 2: "Northeast", 3: "Southeast", 4: "South",
               5: "Central-West"}
UF_REG = {11: "North", 12: "North", 13: "North", 14: "North", 15: "North",
          16: "North", 17: "North", 21: "Northeast", 22: "Northeast",
          23: "Northeast", 24: "Northeast", 25: "Northeast", 26: "Northeast",
          27: "Northeast", 28: "Northeast", 29: "Northeast", 31: "Southeast",
          32: "Southeast", 33: "Southeast", 35: "Southeast", 41: "South",
          42: "South", 43: "South", 50: "Central-West", 51: "Central-West",
          52: "Central-West", 53: "Central-West"}

# ------------------------------------------------------------------ HELPERS
def num(s):
    return pd.to_numeric(s, errors="coerce")


def clean(s, miss=()):
    """Converte para número e apaga SOMENTE os códigos de `miss`.
    Nunca use um conjunto global: 8 e 9 são respostas válidas em e22 e em
    número de moradores."""
    s = num(s)
    return s.mask(s.isin(list(miss))) if len(miss) else s


def clean_weight(s):
    """Peso amostral; PNS 2013 usa vírgula decimal; vazio = não selecionada."""
    s = s.astype(str).str.strip().str.replace(",", ".", regex=False)
    s = s.replace({"": np.nan, "nan": np.nan})
    return pd.to_numeric(s, errors="coerce")


def educ_elsi(v):
    """ELSI e22: 1 Nunca estudou; 2-8 = 1ª a 7ª série; 9 = 8ª série (fund. completo);
    10-12 = 1º a 3º do 2º grau; 13 = supletivo; 14-18 = superior ou mais; 99 = NR."""
    if pd.isna(v):
        return np.nan
    v = int(v)
    if EDU_SCHEME == "4":
        if v == 1:
            return EDU_NONE
        if 2 <= v <= 8:
            return EDU_INC
    elif 1 <= v <= 8:
        return EDU_LOW
    if 9 <= v <= 13:
        return EDU_MID
    if 14 <= v <= 18:
        return EDU_HIGH
    return np.nan


def insurance_elsi(s, wave):
    """u1: detecta 0/1 (W2/W3) ou 1/2 (W1 no questionário). Erra se ambíguo."""
    v = clean(s, {9})
    vals = set(v.dropna().unique())
    if vals <= {0, 1}:
        m = {0: "No", 1: "Yes"}
    elif vals <= {1, 2}:
        m = {1: "Yes", 2: "No"}
    else:
        raise ValueError(f"ELSI W{wave}: u1 com valores inesperados {sorted(vals)}")
    print(f"  [u1 W{wave}] valores {sorted(vals)} -> {m}")
    return v.map(m)


def insurance_pns2013(df):
    """I001 inclui odontológico. Plano MÉDICO = I001==1 e NÃO (só um plano
    e esse plano é apenas odontológico)."""
    i001, i002, i003 = num(df["I001"]), num(df["I002"]), num(df["I003"])
    out = pd.Series(np.nan, index=df.index, dtype=object)
    out[i001 == 2] = "No"
    dental_only = (i001 == 1) & (i003 == 1) & (i002 == 1)
    unknown     = (i001 == 1) & (i003 == 1) & i002.isna()
    out[(i001 == 1) & ~dental_only & ~unknown] = "Yes"
    out[dental_only] = "No"
    return out


# ------------------------------------------------------------------ BUILDERS
def build_elsi(df, wave):
    o = pd.DataFrame(index=df.index)
    o["wave"]       = wave
    o["age"]        = num(df["idade"])
    o["sex"]        = num(df["sexo"])
    o["ethnicity"]  = clean(df["e9"], {9}).map(ETHNIC_ELSI)
    o["marital"]    = clean(df["e7"], {9}).map(MAR_ELSI)
    o["zone"]       = num(df["zona"]).map(ZONE_R)
    o["region"]     = num(df["regiao"]).map(REG_ELSI)
    o["education"]  = num(df["e22"]).apply(educ_elsi)
    o["insurance"]  = insurance_elsi(df["u1"], wave)
    res = "ar6" if wave == 1 else "nmoradores"
    o["n_residents"] = num(df[res]).clip(upper=10)        # sem apagar 8/9
    o["weight"]     = clean_weight(df[ELSI_WEIGHT])

    m13 = num(df["m13"])
    o["target_pap"] = m13.isin(PAP_NONADH_ELSI).astype("Int64")
    o["target_pap"] = o["target_pap"].where(m13.isin(PAP_VALID_ELSI), np.nan)

    m15 = num(df["m15"])
    o["target_mamo"] = m15.isin(MAMO_NONADH_ELSI[wave]).astype("Int64")
    o["target_mamo"] = o["target_mamo"].where(m15.isin(MAMO_VALID_ELSI), np.nan)
    return o[(o["sex"] == 0) & (o["age"] >= AGE_MIN)].copy()


def build_pns(df, year):
    o = pd.DataFrame(index=df.index)
    o["year"]       = year
    o["age"]        = num(df["C008"])
    o["sex"]        = num(df["C006"])
    o["ethnicity"]  = clean(df["C009"], {9}).map(ETHNIC_PNS)
    o["zone"]       = num(df["V0026"]).map(ZONE_R)
    o["region"]     = num(df["V0001"]).map(UF_REG)
    o["education"]  = num(df["VDD004A"]).map(EDUC_PNS)
    nr = num(df["V0022"]); o["n_residents"] = nr.where(nr >= 1).clip(upper=10)  # 0 moradores = inválido (10 casos PNS 2013)
    o["weight"]     = clean_weight(df[PNS_WEIGHT])
    o["uf"]         = num(df["V0001"]).astype("Int64")

    if year == 2013:
        o["marital"]   = clean(df["C010"], {9}).map(MAR_PNS)
        o["insurance"] = insurance_pns2013(df)
        pap_col, mamo_col = "R001", "R017"
    else:
        o["marital"]   = clean(df["C01001"], {9}).map(MAR_PNS)
        o["insurance"] = clean(df["I00102"], {9}).map(INS_PNS)
        pap_col, mamo_col = "R00101", "R01701"

    pap = clean(df[pap_col], {9})
    o["target_pap"] = pap.isin(PAP_NONADH_PNS).astype("Int64")
    o["target_pap"] = o["target_pap"].where(pap.isin(PAP_VALID_PNS), np.nan)
    hyst = clean(df["R010"], {9})                          # 1 Sim / 2 Não (todas)
    o["target_pap"] = o["target_pap"].where(hyst != 1, np.nan)

    md = clean(df["R015"], {9})
    mw = clean(df[mamo_col], {9})
    valid = (md == 2) | ((md == 1) & mw.isin([1, 2, 3, 4]))
    o["target_mamo"] = ((md == 2) | mw.isin(MAMO_NONADH_PNS)).astype("Int64")
    o["target_mamo"] = o["target_mamo"].where(valid, np.nan)

    o["barrier_pap"]  = num(df["R002"]) if year == 2013 else num(df["R002010"])
    o["barrier_mamo"] = num(df["R016"]) if year == 2013 else np.nan
    return o[(o["sex"] == 2) & (o["age"] >= AGE_MIN)].copy()


# ------------------------------------------------------------------ READERS
def read_elsi(path, wave):
    import pyreadstat
    df, _ = pyreadstat.read_dta(str(path), encoding="latin1")
    return build_elsi(df, wave)


PNS_COLS = {"C006","C008","C009","C010","C01001","V0026","V0001","VDD004A","V0022",
            "V00291","I001","I002","I003","I00102","R001","R00101","R010","R015",
            "R016","R017","R01701","R002","R002010"}


def load_pns_raw(path, year):
    """Lê só as colunas necessárias, como texto (rápido; evita inferência de tipos)."""
    return pd.read_csv(path, sep=";" if year == 2013 else ",", encoding="latin1",
                       usecols=lambda c: c in PNS_COLS, dtype=str)


def read_pns(path, year):
    return build_pns(load_pns_raw(path, year), year)


# ------------------------------------------------------------------ X, y, w
def pool_elsi_train(*waves):
    """Concatena ondas do ELSI normalizando o peso (média 1 dentro de cada onda)."""
    out = []
    for d in waves:
        d = d.copy()
        m = d["weight"].mean()
        d["weight"] = d["weight"] / m if pd.notna(m) and m > 0 else d["weight"]
        out.append(d)
    return pd.concat(out, ignore_index=True)


def get_Xy(df, target, age_max, use_weights=False):
    """Retorna X, y, w (w=None se use_weights=False).
    Com use_weights=True descarta quem não tem peso."""
    s = df[(df["age"] >= AGE_MIN) & (df["age"] <= age_max)].copy()
    s["age_group"] = pd.cut(
        s["age"], bins=[AGE_MIN - 1, AGE_CUT - 1, age_max],
        labels=[f"{AGE_MIN}-{AGE_CUT-1}", f"{AGE_CUT}-{age_max}"]).astype(str)
    cols = ALL_FEAT + [target] + (["weight"] if use_weights else [])
    sub = s[cols].dropna()
    X = sub[ALL_FEAT].reset_index(drop=True)
    y = sub[target].astype(int).values
    w = sub["weight"].astype(float).values if use_weights else None
    return X, y, w


# ------------------------------------------------------------------ CAMINHOS
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent      # Final_5_samples/
DATA = ROOT / "Data"
DEFAULT_PATHS = {
    "elsi1":   DATA / "ELSI Portugues (1a onda) stata13.dta",
    "elsi2":   DATA / "ELSI Portugues (2a onda) stata13.dta",
    "elsi3":   DATA / "ELSI Portugues (3a onda).dta",
    "pns2013": DATA / "PNS_2013.csv",
    "pns2019": DATA / "pns2019.csv",
}


def add_path_args(ap, extra=()):
    for k, v in DEFAULT_PATHS.items():
        ap.add_argument("--" + k, default=str(v))
    for k, v in extra:
        ap.add_argument("--" + k, default=str(v))
