"""
TESTE RÁPIDO (não salva nada em disco) — escolaridade 3 vs 4 categorias.
Rode no VS Code (F5 / Run Python File) a partir da pasta Codes/Python.

O que faz
  Para cada esquema de escolaridade ("3" principal; "4" sensibilidade) e cada desfecho
  (Papanicolaou, mamografia): treina em ELSI W1+W2, usa limiar de Youden do ELSI W3 e avalia em
  ELSI W3, PNS 2013 e PNS 2019. Hiperparâmetros FIXOS (sem busca) -> roda em poucos minutos.
  Mostra no console: n, prevalência, AUC, sens/espec, importância relativa por variável (ganho
  do XGBoost agregado por variável original) e % de não adesão por escolaridade.
Obs.: números são para COMPARAR os esquemas, não substituem o pipeline completo.
"""
import warnings; warnings.filterwarnings("ignore")
import os, importlib, sys
import numpy as np, pandas as pd

# --- caminhos (ajuste se necessário). Padrão: Final_5_samples/Data/ ---------------------------
import common as c0
P = c0.DEFAULT_PATHS
# P = {"elsi1": r"...", "elsi2": r"...", "elsi3": r"...", "pns2013": r"...", "pns2019": r"..."}

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from xgboost import XGBClassifier
import pyreadstat

# --- leitura bruta UMA vez ---------------------------------------------------------------------
print("Lendo dados brutos (uma vez)...")
raw_elsi = {w: pyreadstat.read_dta(str(P[k]), encoding="latin1")[0]
            for w, k in [(1, "elsi1"), (2, "elsi2"), (3, "elsi3")]}
raw_pns = {y: c0.load_pns_raw(P[k], y) for y, k in [(2013, "pns2013"), (2019, "pns2019")]}


def make_pre(c):
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    num = Pipeline([("imp", SimpleImputer(strategy="median"))])
    return ColumnTransformer([("cat", cat, c.CAT_FEAT), ("num", num, c.NUM_FEAT)])


def youden(y, p):
    f, t, th = roc_curve(y, p); return float(th[np.argmax(t - f)])


def sens_spec(y, p, thr):
    tn, fp, fn, tp = confusion_matrix(y, (p >= thr).astype(int)).ravel()
    return tp / (tp + fn), tn / (tn + fp)


def run(scheme):
    os.environ["EDU_SCHEME"] = scheme; os.environ.pop("OUT_NAME", None)
    c = importlib.reload(c0)                      # recarrega regras com o esquema escolhido
    w1, w2, w3 = (c.build_elsi(raw_elsi[i].copy(), i) for i in (1, 2, 3))
    train = c.pool_elsi_train(w1, w2)
    sets = {"ELSI W3": w3,
            "PNS 2013": c.build_pns(raw_pns[2013].copy(), 2013),
            "PNS 2019": c.build_pns(raw_pns[2019].copy(), 2019)}
    out = {}
    for name, tgt, amax in [("Pap", "target_pap", c.PAP_MAX_AGE), ("Mamo", "target_mamo", c.MAMO_MAX_AGE)]:
        Xtr, ytr, _ = c.get_Xy(train, tgt, amax)
        res = {"n_train": len(ytr), "prev_train": ytr.mean()}
        # descritivo: % não adesão por escolaridade no treino
        d = Xtr.assign(y=ytr).groupby("education")["y"].agg(["mean", "size"])
        res["edu_table"] = d
        models = {
            "LR": Pipeline([("pre", make_pre(c)), ("clf", LogisticRegression(
                penalty="elasticnet", solver="saga", C=0.1, l1_ratio=0.5, max_iter=3000,
                class_weight="balanced", random_state=42))]),
            "XGB": Pipeline([("pre", make_pre(c)), ("clf", XGBClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05, eval_metric="logloss",
                scale_pos_weight=(ytr == 0).sum() / (ytr == 1).sum(), random_state=42, n_jobs=-1))]),
        }
        data = {k: c.get_Xy(d_, tgt, amax)[:2] for k, d_ in sets.items()}
        for m, pipe in models.items():
            pipe.fit(Xtr, ytr)
            Xv, yv = data["ELSI W3"]
            thr = youden(yv, pipe.predict_proba(Xv)[:, 1])
            for k, (X, y) in data.items():
                p = pipe.predict_proba(X)[:, 1]
                se, sp = sens_spec(y, p, thr)
                res[(m, k)] = dict(n=len(y), prev=y.mean(), auc=roc_auc_score(y, p), sens=se, spec=sp)
        # importância por variável (ganho XGBoost agregado)
        pipe = models["XGB"]
        ohe = pipe["pre"].named_transformers_["cat"]["ohe"]
        names = ohe.get_feature_names_out(c.CAT_FEAT).tolist() + c.NUM_FEAT
        gain = pipe["clf"].get_booster().get_score(importance_type="total_gain")
        imp = np.array([gain.get(f"f{i}", 0.0) for i in range(len(names))])
        var = [next((v for v in c.CAT_FEAT if n.startswith(v + "_")), n) for n in names]
        s = pd.Series(imp, index=var).groupby(level=0).sum()
        res["importance"] = (s / s.sum() * 100).sort_values(ascending=False)
        out[name] = res
    return out


results = {s: run(s) for s in ("3", "4")}

# --- relatório -----------------------------------------------------------------------------------
pd.set_option("display.width", 160)
for outc in ("Pap", "Mamo"):
    print("\n" + "=" * 92 + f"\n{outc.upper()}\n" + "=" * 92)
    rows = []
    for s in ("3", "4"):
        for m in ("LR", "XGB"):
            for k in ("ELSI W3", "PNS 2013", "PNS 2019"):
                r = results[s][outc][(m, k)]
                rows.append(dict(educ=f"{s} cat", modelo=m, amostra=k, n=r["n"],
                                 prev=f"{r['prev']:.1%}", AUC=round(r["auc"], 3),
                                 sens=round(r["sens"], 2), espec=round(r["spec"], 2)))
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nImportância relativa (% do ganho do XGBoost, por variável):")
    imp = pd.concat({f"{s} cat": results[s][outc]["importance"] for s in ("3", "4")}, axis=1).round(1)
    print(imp.sort_values("3 cat", ascending=False).to_string())
    for s in ("3", "4"):
        t = results[s][outc]["edu_table"].copy()
        t["mean"] = (t["mean"] * 100).round(1)
        print(f"\n% não adesão por escolaridade — treino ELSI (esquema {s} cat):")
        print(t.rename(columns={"mean": "% não adesão", "size": "n"}).to_string())
print("\nFim. Nada foi salvo em disco.")
