"""
TESTE DE ROBUSTEZ — a região pesa mais na mamografia do que no Papanicolaou?
Não grava nada em disco: só imprime. Rode no VS Code a partir de Codes/Python.

Usa os modelos JÁ TREINADOS em Outputs_v3_edu4/Models (4 categorias de escolaridade).
Para cada exame, algoritmo (LogReg, RandomForest, XGBoost) e amostra (ELSI W3, PNS 2013, PNS 2019):
  importância por PERMUTAÇÃO de cada variável = queda do AUC quando a variável é embaralhada
  (K permutações; IC 95% por bootstrap das mulheres).
Ao final: contraste de região (mamografia − Papanicolaou) com IC.
"""
import warnings; warnings.filterwarnings("ignore")
import os
os.environ["EDU_SCHEME"] = "4"; os.environ["OUT_NAME"] = "Outputs_v3_edu4"
import numpy as np, pandas as pd, joblib, pyreadstat
import common as c
from sklearn.metrics import roc_auc_score

K      = int(os.environ.get("K_PERM", 20))      # permutações por variável
B      = int(os.environ.get("N_BOOT", 300))     # reamostragens bootstrap
VARS   = c.ALL_FEAT
MODELS = c.ROOT / c.OUT_NAME / "Models"
P      = c.DEFAULT_PATHS
rng    = np.random.default_rng(42)

print("Lendo dados...");
w3   = c.build_elsi(pyreadstat.read_dta(str(P["elsi3"]), encoding="latin1")[0], 3)
pns  = {2013: c.read_pns(P["pns2013"], 2013), 2019: c.read_pns(P["pns2019"], 2019)}
sets = {"ELSI W3": w3, "PNS 2013": pns[2013], "PNS 2019": pns[2019]}
OUT  = {"Pap smear": ("pap_smear", "target_pap", c.PAP_MAX_AGE),
        "Mammography": ("mammography", "target_mamo", c.MAMO_MAX_AGE)}
ALG  = ["LogReg", "RandomForest", "XGBoost"]


def perm_drops(pipe, X, y):
    """Retorna dict var -> array (B,) com a queda de AUC por reamostra bootstrap."""
    n = len(y); p0 = pipe.predict_proba(X)[:, 1]
    idx_b = [rng.integers(0, n, n) for _ in range(B)]
    base = np.array([roc_auc_score(y[i], p0[i]) for i in idx_b])
    res = {}
    for v in VARS:
        pk = []
        for _ in range(K):
            Xp = X.copy(); Xp[v] = rng.permutation(Xp[v].values)
            pk.append(pipe.predict_proba(Xp)[:, 1])
        drop = np.array([base[b] - np.mean([roc_auc_score(y[i], p[i]) for p in pk])
                         for b, i in enumerate(idx_b)])
        res[v] = drop
    return res


store = {}
for outc, (stem, tgt, amax) in OUT.items():
    for alg in ALG:
        pipe = joblib.load(MODELS / f"model_{stem}_{alg}.joblib")
        for ds, df in sets.items():
            X, y, _ = c.get_Xy(df, tgt, amax)
            print(f"  {outc} | {alg} | {ds} (n={len(y)})", flush=True)
            store[(outc, alg, ds)] = perm_drops(pipe, X, y)

pd.set_option("display.width", 170)
print("\n" + "=" * 100)
print("QUEDA DO AUC (x1000) ao embaralhar a variável — média [IC 95%]. Maior = variável mais importante.")
print("=" * 100)
for ds in sets:
    for alg in ALG:
        rows = {}
        for outc in OUT:
            d = store[(outc, alg, ds)]
            for v in VARS:
                m = d[v].mean() * 1000; lo, hi = np.percentile(d[v], [2.5, 97.5]) * 1000
                rows.setdefault(v, {})[outc] = f"{m:5.1f} [{lo:4.1f},{hi:4.1f}]"
        t = pd.DataFrame(rows).T
        order = sorted(VARS, key=lambda v: -store[("Mammography", alg, ds)][v].mean())
        print(f"\n{ds} | {alg}"); print(t.loc[order].to_string())

print("\n" + "=" * 100)
print("CONTRASTE DE REGIÃO: queda de AUC (x1000) na mamografia MENOS no Papanicolaou  [IC 95%]")
print("  >0 e IC sem zero  => região pesa mais na mamografia")
print("=" * 100)
rows = []
for ds in sets:
    for alg in ALG:
        dm = store[("Mammography", alg, ds)]["region"]; dp = store[("Pap smear", alg, ds)]["region"]
        diff = (dm - rng.permutation(dp)) * 1000          # bootstraps independentes (amostras diferentes)
        lo, hi = np.percentile(diff, [2.5, 97.5])
        rows.append(dict(amostra=ds, algoritmo=alg, mamo=round(dm.mean() * 1000, 1),
                         pap=round(dp.mean() * 1000, 1), diferenca=round(diff.mean(), 1),
                         IC95=f"[{lo:.1f}, {hi:.1f}]", sem_zero="SIM" if lo > 0 or hi < 0 else "não"))
print(pd.DataFrame(rows).to_string(index=False))

print("\nRanking da região (1 = mais importante entre as 8 variáveis), média das 3 amostras:")
rk = []
for outc in OUT:
    for alg in ALG:
        r = [sorted(VARS, key=lambda v: -store[(outc, alg, ds)][v].mean()).index("region") + 1 for ds in sets]
        rk.append(dict(exame=outc, algoritmo=alg, ranking_regiao=round(np.mean(r), 1)))
print(pd.DataFrame(rk).to_string(index=False))
print("\nFim. Nada foi salvo em disco.")
