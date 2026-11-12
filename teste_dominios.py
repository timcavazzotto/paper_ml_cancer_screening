"""
TESTE PRÉ-ESPECIFICADO (22/10/2026) — importância por DOMÍNIO e contraste entre exames.
Não grava nada em disco: só imprime. Rode no VS Code a partir de Codes/Python.

DOMÍNIOS (teoria, modelo de Andersen; não mudam depois dos resultados)
  D1 Acesso e oferta        : region, zone, insurance
  D2 Posição socioeconômica : education
  D3 Apoio social/domicílio : marital, n_residents
  Controles                 : age_group, ethnicity
HIPÓTESES DIRECIONAIS (contraste = mamografia − Papanicolaou)
  H1  D1 pesa mais na mamografia   (contraste > 0)
  H2  D2 pesa mais no Papanicolaou (contraste < 0)
  H3  D3 pesa mais no Papanicolaou (contraste < 0)
REGRA: uma hipótese é APOIADA se (Teste 1, XGBoost) o contraste tem o mesmo sinal nas 3 amostras e o
IC de 98,3% (Bonferroni, 3 hipóteses) exclui zero em >=2 delas, E (Teste 2) o contraste tem o mesmo
sinal em >=2 das 3 amostras.  Tudo é impresso, apoiado ou não.
SENSIBILIDADE única: plano de saúde movido de D1 para D2.

TESTE 1  importância por permutação em BLOCO (queda de AUC), modelos já treinados, 3 amostras
TESTE 2  regressão logística refeita no treino ELSI: queda de AUC ao RETIRAR o domínio (fora da amostra)
         e AUC do modelo só com o domínio
TESTE 3  SHAP agrupado por domínio (XGBoost), nas 3 amostras
"""
import warnings; warnings.filterwarnings("ignore")
import os
os.environ["EDU_SCHEME"] = "4"; os.environ["OUT_NAME"] = "Outputs_v3_edu4"
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import common as c

K = int(os.environ.get("K_PERM", 8)); B = int(os.environ.get("N_BOOT", 100))
ALGS = os.environ.get("ALGS", "XGBoost,LogReg").split(",")
LO_Q, HI_Q = 0.85, 99.15                         # IC 98,3%
rng = np.random.default_rng(42)

DOM = {"D1 Acesso": ["region", "zone", "insurance"],
       "D2 Socioeconômico": ["education"],
       "D3 Apoio social": ["marital", "n_residents"],
       "Controles": ["age_group", "ethnicity"]}
DOM_SENS = {"D1' (sem plano)": ["region", "zone"],
            "D2' (+plano)": ["education", "insurance"],
            "D3 Apoio social": ["marital", "n_residents"],
            "Controles": ["age_group", "ethnicity"]}
HYP = {"D1 Acesso": ("H1", +1), "D2 Socioeconômico": ("H2", -1), "D3 Apoio social": ("H3", -1)}


# ------------------------------------------------------------------ Teste 1
def block_drops(pipe, X, y, domains):
    """dict domínio -> array (B,) de queda de AUC (bootstrap das mulheres, K permutações em bloco)."""
    n = len(y); p0 = pipe.predict_proba(X)[:, 1]
    idx = [rng.integers(0, n, n) for _ in range(B)]
    base = np.array([roc_auc_score(y[i], p0[i]) for i in idx])
    out = {}
    for name, cols in domains.items():
        pk = []
        for _ in range(K):
            Xp = X.copy(); perm = rng.permutation(n)
            Xp[cols] = X[cols].values[perm]               # embaralha o bloco inteiro junto
            pk.append(pipe.predict_proba(Xp)[:, 1])
        out[name] = np.array([base[b] - np.mean([roc_auc_score(y[i], p[i]) for p in pk])
                              for b, i in enumerate(idx)])
    return out


def contrast(dm, dp):
    d = (dm - dp) * 1000
    return d.mean(), np.percentile(d, LO_Q), np.percentile(d, HI_Q)


# ------------------------------------------------------------------ Teste 2
def design(frames, cols):
    big = pd.concat([f.assign(_k=k) for k, f in frames.items()], ignore_index=True)
    catc = [v for v in cols if v != "n_residents"]
    Z = pd.get_dummies(big[catc], drop_first=True).astype(float) if catc else pd.DataFrame(index=big.index)
    if "n_residents" in cols:
        Z["n_residents"] = big["n_residents"].astype(float)
    Z = (Z - Z.mean()) / Z.std().replace(0, 1)
    return {k: Z[big["_k"] == k].reset_index(drop=True) for k in frames}


def fit_pred(frames, ytr, cols):
    D = design(frames, cols)
    m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000).fit(D["train"], ytr)
    return {k: m.predict_proba(D[k])[:, 1] for k in frames if k != "train"}


def drop_domain(data, domains):
    """Para cada amostra de validação: dict domínio -> (queda_AUC_bootstrap (B,), AUC só-domínio)."""
    Xtr, ytr = data["train"]
    frames = {k: v[0] for k, v in data.items()}
    full = fit_pred(frames, ytr, c.ALL_FEAT)
    res = {k: {} for k in frames if k != "train"}
    for name, cols in domains.items():
        rest = [v for v in c.ALL_FEAT if v not in cols]
        pr = fit_pred(frames, ytr, rest); po = fit_pred(frames, ytr, cols)
        for k in res:
            y = data[k][1]; n = len(y)
            idx = [rng.integers(0, n, n) for _ in range(B)]
            dr = np.array([roc_auc_score(y[i], full[k][i]) - roc_auc_score(y[i], pr[k][i]) for i in idx])
            res[k][name] = (dr, roc_auc_score(y, po[k]))
    return res


# ------------------------------------------------------------------ Teste 3
def shap_domain(pipe, X, domains):
    import shap
    ohe = pipe["pre"].named_transformers_["cat"]["ohe"]
    names = ohe.get_feature_names_out(c.CAT_FEAT).tolist() + c.NUM_FEAT
    var = [next((v for v in c.CAT_FEAT if nm.startswith(v + "_")), nm) for nm in names]
    S = np.asarray(shap.TreeExplainer(pipe["clf"]).shap_values(np.asarray(pipe["pre"].transform(X))))
    cols = {d: np.abs(S[:, [i for i, v in enumerate(var) if v in vs]].sum(axis=1)) for d, vs in domains.items()}
    return pd.DataFrame(cols)                                 # |SHAP do domínio| por mulher


def share_boot(df):
    n = len(df); a = df.values
    sh = []
    for _ in range(B):
        m = a[rng.integers(0, n, n)].mean(axis=0); sh.append(m / m.sum() * 100)
    sh = np.array(sh); base = a.mean(axis=0); return base / base.sum() * 100, sh


def main():
    import pyreadstat, joblib
    P = c.DEFAULT_PATHS
    el = {w: c.build_elsi(pyreadstat.read_dta(str(P[k]), encoding="latin1")[0], w)
          for w, k in [(1, "elsi1"), (2, "elsi2"), (3, "elsi3")]}
    sets = {"train": c.pool_elsi_train(el[1], el[2]), "ELSI W3": el[3],
            "PNS 2013": c.read_pns(P["pns2013"], 2013), "PNS 2019": c.read_pns(P["pns2019"], 2019)}
    VAL = ["ELSI W3", "PNS 2013", "PNS 2019"]
    OUT = {"Pap smear": ("target_pap", c.PAP_MAX_AGE, "pap_smear"),
           "Mammography": ("target_mamo", c.MAMO_MAX_AGE, "mammography")}
    data = {o: {k: c.get_Xy(d, t, a)[:2] for k, d in sets.items()} for o, (t, a, _) in OUT.items()}
    M = c.ROOT / c.OUT_NAME / "Models"
    pd.set_option("display.width", 170)
    verdict = {}

    for label, domains in [("PRINCIPAL", DOM), ("SENSIBILIDADE (plano em D2)", DOM_SENS)]:
        print("\n" + "=" * 100 + f"\nTESTE 1 — queda de AUC (x1000) por permutação em bloco — {label}\n" + "=" * 100)
        t1 = {}
        for o, (_, _, stem) in OUT.items():
            for alg in ALGS:
                pipe = joblib.load(M / f"model_{stem}_{alg}.joblib")
                for ds in VAL:
                    X, y = data[o][ds]; print(f"  {o} | {alg} | {ds} (n={len(y)})", flush=True)
                    t1[(o, alg, ds)] = block_drops(pipe, X, y, domains)
        for alg in ALGS:
            rows = []
            for d in domains:
                for ds in VAL:
                    dm, dp = t1[("Mammography", alg, ds)][d], t1[("Pap smear", alg, ds)][d]
                    m, lo, hi = contrast(dm, dp)
                    rows.append(dict(dominio=d, amostra=ds, mamo=round(dm.mean() * 1000, 1),
                                     pap=round(dp.mean() * 1000, 1), contraste=round(m, 1),
                                     IC98=f"[{lo:.1f}, {hi:.1f}]", exclui_zero="SIM" if lo > 0 or hi < 0 else "não"))
            print(f"\n{alg}:"); print(pd.DataFrame(rows).to_string(index=False))
        if label == "PRINCIPAL":
            verdict["t1"] = {(d, ds): contrast(t1[("Mammography", "XGBoost", ds)][d], t1[("Pap smear", "XGBoost", ds)][d])
                             for d in domains for ds in VAL if "XGBoost" in ALGS}

    print("\n" + "=" * 100 + "\nTESTE 2 — regressão logística (treino ELSI W1+W2): queda de AUC ao RETIRAR o domínio (x1000)"
          "\n          e AUC do modelo só com o domínio, fora da amostra\n" + "=" * 100)
    t2 = {o: drop_domain(data[o], DOM) for o in OUT}
    rows = []
    for d in DOM:
        for ds in VAL:
            dm, am = t2["Mammography"][ds][d]; dp, ap = t2["Pap smear"][ds][d]
            m, lo, hi = contrast(dm, dp)
            rows.append(dict(dominio=d, amostra=ds, queda_mamo=round(dm.mean() * 1000, 1), queda_pap=round(dp.mean() * 1000, 1),
                             contraste=round(m, 1), IC98=f"[{lo:.1f}, {hi:.1f}]",
                             AUC_so_dominio_mamo=round(am, 3), AUC_so_dominio_pap=round(ap, 3)))
            verdict.setdefault("t2", {})[(d, ds)] = m
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 100 + "\nTESTE 3 — SHAP agrupado por domínio (XGBoost): % do |SHAP| total [IC 95% bootstrap]\n" + "=" * 100)
    for ds in VAL:
        rows = {}
        for o, (_, _, stem) in OUT.items():
            pipe = joblib.load(M / f"model_{stem}_XGBoost.joblib")
            X, y = data[o][ds]; b, sh = share_boot(shap_domain(pipe, X, DOM))
            for j, d in enumerate(DOM):
                rows.setdefault(d, {})[o] = f"{b[j]:5.1f} [{np.percentile(sh[:, j], 2.5):4.1f},{np.percentile(sh[:, j], 97.5):4.1f}]"
        print(f"\n{ds}"); print(pd.DataFrame(rows).T.to_string())

    print("\n" + "=" * 100 + "\nVEREDITO PRÉ-ESPECIFICADO (regra: Teste 1 XGBoost + Teste 2)\n" + "=" * 100)
    for d, (h, sgn) in HYP.items():
        if "t1" not in verdict: break
        s1 = [verdict["t1"][(d, ds)] for ds in VAL]
        same = all(np.sign(m) == sgn for m, _, _ in s1)
        ex = sum(1 for m, lo, hi in s1 if (lo > 0 if sgn > 0 else hi < 0))
        s2 = sum(1 for ds in VAL if np.sign(verdict["t2"][(d, ds)]) == sgn)
        ok = same and ex >= 2 and s2 >= 2
        print(f"  {h} ({d}, esperado {'mamo>pap' if sgn > 0 else 'pap>mamo'}): sinal igual nas 3 amostras={'sim' if same else 'não'}; "
              f"IC exclui zero em {ex}/3; Teste 2 mesmo sinal em {s2}/3  ->  {'APOIADA' if ok else 'NÃO apoiada'}")
    print("\nFim. Nada foi salvo em disco.")


if __name__ == "__main__":
    main()
