"""R1-7 — sensibilidade: excluir mulheres com câncer prévio (ELSI n60==1; PNS Q120==1).
(A) modelo XGBoost JÁ TREINADO (congelado) reavaliado nas validações sem essas mulheres;
(B) XGBoost REAJUSTADO (mesmos hiperparâmetros) no treino sem essas mulheres e avaliado nas validações sem elas.
Colocar em Codes/Python. Requer Models/model_<desfecho>_XGBoost.joblib. Saída: Python_tables/prior_cancer_sensitivity.csv"""
import os, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("EDU_SCHEME", "4"); os.environ.setdefault("OUT_NAME", "Outputs_v3_edu4")
import numpy as np, pandas as pd, joblib, pyreadstat
from pathlib import Path
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from common_legacy import *

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_TAB = ROOT / OUT_NAME / "Python_tables"; MODELS = ROOT / OUT_NAME / "Models"
N_BOOT = 1000; rng = np.random.default_rng(42)

def with_flag(raw, built, col):
    built = built.copy()
    built["prior_ca"] = (num(raw[col]).loc[built.index] == 1).values
    return built

def load_all():
    d = {}
    for w, p in [(1, ELSI_W1), (2, ELSI_W2), (3, ELSI_W3)]:
        raw, _ = pyreadstat.read_dta(str(p), encoding="latin1")
        d[f"W{w}"] = with_flag(raw, build_elsi(raw, w), "n60")
    for y, p in [(2013, PNS13), (2019, PNS19)]:
        cols = PNS_COLS | {"Q120"}
        raw = pd.read_csv(p, sep=";" if y == 2013 else ",", encoding="latin1",
                          usecols=lambda c: c in cols, dtype=str)
        d[f"PNS{y}"] = with_flag(raw, build_pns(raw, y), "Q120")
    return d

def get_Xy(df, target, amax):
    s = df[(df["age"] >= 50) & (df["age"] <= amax)].copy()
    s["age_group"] = pd.cut(s["age"], bins=[49, 59, amax], labels=["50-59", f"60-{amax}"]).astype(str)
    sub = s[ALL_FEAT + [target, "prior_ca"]].dropna()
    return sub[ALL_FEAT].reset_index(drop=True), sub[target].astype(int).values, sub["prior_ca"].values

def auc_ci(y, p):
    a = roc_auc_score(y, p); n = len(y); bs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if y[i].min() != y[i].max(): bs.append(roc_auc_score(y[i], p[i]))
    return a, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

D = load_all(); train = pd.concat([D["W1"], D["W2"]], ignore_index=True)
rows = []
for outcome, target, amax in [("pap_smear", "target_pap", 64), ("mammography", "target_mamo", 69)]:
    frozen = joblib.load(MODELS / f"model_{outcome}_XGBoost.joblib")
    Xtr, ytr, ftr = get_Xy(train, target, amax)
    refit = clone(frozen)
    keep = ~ftr
    refit.set_params(clf__scale_pos_weight=(ytr[keep] == 0).sum() / max((ytr[keep] == 1).sum(), 1))
    refit.fit(Xtr[keep], ytr[keep])
    rows.append(dict(outcome=outcome, sample="ELSI W1+W2 (train)", n_all=len(ytr), n_excl=int(keep.sum()),
                     n_prior_cancer=int(ftr.sum()), note="refit excludes these women"))
    for name, df in [("ELSI W3", D["W3"]), ("PNS 2013", D["PNS2013"]), ("PNS 2019", D["PNS2019"])]:
        X, y, f = get_Xy(df, target, amax); k = ~f
        pa = frozen.predict_proba(X)[:, 1]
        a_all = auc_ci(y, pa)
        a_ex = auc_ci(y[k], pa[k])
        a_re = auc_ci(y[k], refit.predict_proba(X[k])[:, 1])
        for lab, a in [("A frozen, all women", a_all), ("A frozen, excl. prior cancer", a_ex),
                       ("B refit w/o prior cancer, excl. prior cancer", a_re)]:
            rows.append(dict(outcome=outcome, sample=name, analysis=lab, n=len(y) if "all" in lab else int(k.sum()),
                             n_prior_cancer=int(f.sum()), AUC=round(a[0], 3), lo=round(a[1], 3), hi=round(a[2], 3)))
res = pd.DataFrame(rows); res.to_csv(OUT_TAB / "prior_cancer_sensitivity.csv", index=False)
print(res.to_string())
