"""
concordancia_modelos.py — AS EXPLICAÇÕES DEPENDEM DO ALGORITMO?  (resposta ao revisor)
=====================================================================================
O XGBoost é o modelo primário (pré-especificado). Aqui as MESMAS explicações são repetidas nos três
algoritmos já treinados (XGBoost, Random Forest, Regressão logística) e a concordância é quantificada.

O que é calculado (todos os modelos são os .joblib já treinados; nada é reajustado, exceto "domínio sozinho")
  1. SHAP por variável (ELSI W3)       XGBoost/RF: TreeExplainer | LogReg: SHAP linear exato = coef x (x - média)
                                       -> importância = média de |soma do SHAP das colunas da variável| (% do total)
                                       -> IC 95% por bootstrap das mulheres (as MESMAS mulheres nos 3 modelos)
                                       -> Spearman entre rankings, sobreposição top-3/top-5, concordância de direção por nível
  2. Permutação por variável           queda de AUC (x1000), 3 amostras de validação
  3. Permutação em bloco por domínio   queda de AUC (x1000), 3 amostras de validação
  4. "Domínio sozinho"                 AUC fora da amostra de um modelo treinado só com as variáveis do domínio
                                       (hiperparâmetros do modelo final de cada algoritmo; treino ELSI W1+W2)

Por padrão XGBoost e LogReg REAPROVEITAM variable_importance_perm.csv e domain_importance_block.csv (já calculados em
analise_dominios_v3.py) e só a Random Forest é calculada. Para recalcular os três:  $env:RECOMPUTE_ALL="1"

Execução (a partir de Codes/Python):
    python concordancia_modelos.py                        (RF: ~15-30 min com K_PERM=6, N_BOOT=500)
    $env:K_PERM="10"; $env:N_BOOT="1000"; python concordancia_modelos.py     (rodada final)
    $env:SHAP_N="2000"; ...                               (sub-amostra de mulheres do W3 para o SHAP; padrão 0 = todas)

Saídas (Outputs_v3_edu4/Python_tables):
    concordance_shap_variable.csv   share (%) do |SHAP| por variável, modelo, exame (+ IC 95%) e posto
    concordance_shap_levels.csv     efeito médio por categoria (e sinal) em cada modelo
    concordance_summary.csv         Spearman (IC), top-3/top-5, concordância de direção, por análise/exame/amostra
    concordance_perm_variable.csv   permutação por variável (3 algoritmos x 3 amostras)
    concordance_perm_domain.csv     permutação em bloco por domínio (3 algoritmos x 3 amostras)
    domain_alone.csv                AUC do domínio sozinho (3 algoritmos x 3 amostras; IC 95% bootstrap)
    concordance_shap_cache.pkl      matrizes SHAP (para os beeswarms suplementares)
"""
import warnings; warnings.filterwarnings("ignore")
import os
os.environ["EDU_SCHEME"] = "4"; os.environ["OUT_NAME"] = "Outputs_v3_edu4"
os.environ.setdefault("K_PERM", "6"); os.environ.setdefault("N_BOOT", "500")
import pickle, time
import numpy as np, pandas as pd
from itertools import combinations
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import common as c
import teste_dominios as td

ALGS = ["XGBoost", "RandomForest", "LogReg"]
VAL = ["ELSI W3", "PNS 2013", "PNS 2019"]
OUTC = {"Pap smear": ("target_pap", c.PAP_MAX_AGE, "pap_smear"),
        "Mammography": ("target_mamo", c.MAMO_MAX_AGE, "mammography")}
DOM = {"Access": ["region", "zone", "insurance"], "Socioeconomic": ["education"],
       "Social support": ["marital", "n_residents"], "Controls": ["age_group", "ethnicity"]}
VARS = c.ALL_FEAT
B_SHAP = 500
rng = np.random.default_rng(2026)


def var_of(name):
    return next((v for v in c.CAT_FEAT if name.startswith(v + "_")), name)


# ------------------------------------------------------------------ SHAP nos três algoritmos
def shap_matrix(pipe, X, alg):
    """Matriz (n x p) de SHAP nas colunas do pré-processador. XGBoost: log-odds; RF: probabilidade; LogReg: log-odds."""
    import shap
    M = np.asarray(pipe["pre"].transform(X)).astype(float)
    clf = pipe["clf"]
    if alg == "LogReg":
        S = (M - M.mean(axis=0)) * clf.coef_.ravel()           # SHAP exato de um modelo linear (preditores independentes)
    else:
        S = shap.TreeExplainer(clf).shap_values(M)
        if isinstance(S, list):
            S = S[1]
        S = np.asarray(S)
        if S.ndim == 3:
            S = S[:, :, 1]
    names = pipe["pre"].named_transformers_["cat"]["ohe"].get_feature_names_out(c.CAT_FEAT).tolist() + list(c.NUM_FEAT)
    return M, S, names


def variable_matrix(S, names):
    """(n x 8): SHAP somado por variável (com sinal) — mesma unidade usada no domínio."""
    vs = [v for v in VARS if any(var_of(n) == v for n in names)]
    idx = {v: [i for i, n in enumerate(names) if var_of(n) == v] for v in vs}
    return np.column_stack([S[:, idx[v]].sum(axis=1) for v in vs]), vs


def shares(V):
    m = np.abs(V).mean(axis=0)
    return m / m.sum() * 100


def top_overlap(a, b, k):
    return len(set(np.argsort(-a)[:k]) & set(np.argsort(-b)[:k])) / k


def level_table(M, S, names, n_min=30):
    """Efeito médio do SHAP entre as mulheres que POSSUEM a categoria; n_residents: sinal da correlação valor x SHAP."""
    rows = []
    scale = np.mean(np.abs(variable_matrix(S, names)[0]).mean(axis=0))
    for j, nm in enumerate(names):
        v = var_of(nm)
        if v == "n_residents":
            r = spearmanr(M[:, j], S[:, j])[0]
            rows.append(dict(variable=v, level="(per additional resident)", mean_shap=np.nan,
                             effect_rel=np.nan, sign=int(np.sign(r)) if np.isfinite(r) else 0, n=len(M), prevalence=np.nan))
            continue
        m = M[:, j] == 1
        if m.sum() < n_min:
            continue
        mu = float(S[m, j].mean())
        rows.append(dict(variable=v, level=nm[len(v) + 1:], mean_shap=mu, effect_rel=mu / scale,
                         sign=int(np.sign(mu)), n=int(m.sum()), prevalence=float(m.mean())))
    d = pd.DataFrame(rows)
    # variáveis binárias: as duas colunas são espelhos; mantém só a categoria minoritária (evita contar duas vezes)
    keep = []
    for v, g in d.groupby("variable"):
        if v == "n_residents" or len(g) != 2:
            keep += list(g.index)
        else:
            keep.append(g.prevalence.idxmin())
    return d.loc[sorted(set(keep))].reset_index(drop=True)


# ------------------------------------------------------------------ passo 1: SHAP
def run_shap(data, models, algs=ALGS):
    shap_cache, rows_v, rows_l, rows_s = {}, [], [], []
    n_sub = int(os.environ.get("SHAP_N", "0"))
    for o, (_, _, stem) in OUTC.items():
        X, y = data[o]["ELSI W3"]
        sub = np.arange(len(X))
        if n_sub and n_sub < len(X):
            sub = np.sort(rng.choice(len(X), n_sub, replace=False))
        Xs = X.iloc[sub].reset_index(drop=True)
        Vm, per = {}, {}
        for alg in algs:
            t0 = time.time()
            M, S, names = shap_matrix(models[(o, alg)], Xs, alg)
            V, vs = variable_matrix(S, names)
            Vm[alg] = V
            shap_cache[(o, alg)] = dict(shap=S, X=M, names=names, n=len(Xs))
            lt = level_table(M, S, names); lt.insert(0, "algorithm", alg); lt.insert(0, "outcome", o); rows_l.append(lt)
            print(f"  [SHAP] {o} | {alg}: {time.time() - t0:.0f} s (n={len(Xs)})", flush=True)
        n = len(Xs); idx_b = [rng.integers(0, n, n) for _ in range(B_SHAP)]
        base = {a: shares(Vm[a]) for a in algs}
        boot = {a: np.array([shares(Vm[a][i]) for i in idx_b]) for a in algs}
        for a in algs:
            rk = (-base[a]).argsort().argsort() + 1
            for j, v in enumerate(vs):
                rows_v.append(dict(outcome=o, algorithm=a, variable=v, share_pct=base[a][j], lo=np.percentile(boot[a][:, j], 2.5),
                                   hi=np.percentile(boot[a][:, j], 97.5), rank=int(rk[j])))
        lv = pd.concat([r for r in rows_l if r.outcome.iloc[0] == o]).set_index(["algorithm", "variable", "level"]).sign
        for a1, a2 in combinations(algs, 2):
            rho = spearmanr(base[a1], base[a2])[0]
            rb = np.array([spearmanr(boot[a1][b], boot[a2][b])[0] for b in range(B_SHAP)])
            s1 = lv[a1]; s2 = lv[a2]; both = s1.index.intersection(s2.index)
            rows_s.append(dict(analysis="SHAP (variable)", outcome=o, dataset="ELSI W3", pair=f"{a1} vs {a2}", rho=rho,
                               rho_lo=np.nanpercentile(rb, 2.5), rho_hi=np.nanpercentile(rb, 97.5),
                               top3=top_overlap(base[a1], base[a2], 3), top5=top_overlap(base[a1], base[a2], 5),
                               dir_agree_pct=float((s1[both] == s2[both]).mean() * 100), n_levels=len(both)))
        if len(algs) == 3:                                    # direção igual nos três modelos
            ls = {a: lv[a] for a in algs}; both = ls[algs[0]].index
            for a in algs[1:]:
                both = both.intersection(ls[a].index)
            ok = np.all([ls[a][both].values == ls[algs[0]][both].values for a in algs[1:]], axis=0)
            rows_s.append(dict(analysis="SHAP (variable)", outcome=o, dataset="ELSI W3", pair="all three",
                               rho=np.nan, rho_lo=np.nan, rho_hi=np.nan,
                               top3=len(set.intersection(*[set(np.argsort(-base[a])[:3]) for a in algs])) / 3,
                               top5=len(set.intersection(*[set(np.argsort(-base[a])[:5]) for a in algs])) / 5,
                               dir_agree_pct=float(ok.mean() * 100), n_levels=len(both)))
    return (pd.DataFrame(rows_v), pd.concat(rows_l, ignore_index=True), pd.DataFrame(rows_s), shap_cache)


# ------------------------------------------------------------------ passo 2/3: permutação
def run_perm(data, models, out_tab, algs=ALGS):
    """Variável e domínio (bloco). Reaproveita XGBoost/LogReg dos CSV existentes, a menos que RECOMPUTE_ALL=1."""
    recompute = os.environ.get("RECOMPUTE_ALL", "0") == "1"
    old_v = out_tab / "variable_importance_perm.csv"; old_d = out_tab / "domain_importance_block.csv"
    have = (not recompute) and old_v.exists() and old_d.exists()
    todo = [a for a in algs if (recompute or not have or a == "RandomForest")]
    blocks = {**DOM, **{"var:" + v: [v] for v in VARS}}
    rv, rd = [], []
    if have:
        ov = pd.read_csv(old_v); od = pd.read_csv(old_d)
        od = od[od.set == "principal"] if "set" in od else od
        for a in algs:
            if a in todo:
                continue
            rv.append(ov[ov.algorithm == a][["outcome", "algorithm", "dataset", "variable", "auc_drop_x1000", "lo", "hi"]])
            rd.append(od[od.algorithm == a][["outcome", "algorithm", "dataset", "domain", "auc_drop_x1000", "lo", "hi"]])
    for o, (_, _, stem) in OUTC.items():
        for a in todo:
            for ds in VAL:
                X, y = data[o][ds]; t0 = time.time()
                d = td.block_drops(models[(o, a)], X, y, blocks)
                print(f"  [PERM] {o} | {a} | {ds} (n={len(y)}): {time.time() - t0:.0f} s", flush=True)
                for name, arr in d.items():
                    lo, hi = np.percentile(arr * 1000, [2.5, 97.5]); m = arr.mean() * 1000
                    if name.startswith("var:"):
                        rv.append(pd.DataFrame([dict(outcome=o, algorithm=a, dataset=ds, variable=name[4:], auc_drop_x1000=m, lo=lo, hi=hi)]))
                    else:
                        rd.append(pd.DataFrame([dict(outcome=o, algorithm=a, dataset=ds, domain=name, auc_drop_x1000=m, lo=lo, hi=hi)]))
    pv = pd.concat(rv, ignore_index=True); pdm = pd.concat(rd, ignore_index=True)
    rows = []
    for o in OUTC:
        for ds in VAL:
            for a1, a2 in combinations(algs, 2):
                x1 = pv[(pv.outcome == o) & (pv.dataset == ds) & (pv.algorithm == a1)].set_index("variable").auc_drop_x1000
                x2 = pv[(pv.outcome == o) & (pv.dataset == ds) & (pv.algorithm == a2)].set_index("variable").auc_drop_x1000.reindex(x1.index)
                rows.append(dict(analysis="Permutation (variable)", outcome=o, dataset=ds, pair=f"{a1} vs {a2}",
                                 rho=spearmanr(x1, x2)[0], rho_lo=np.nan, rho_hi=np.nan,
                                 top3=top_overlap(x1.values, x2.values, 3), top5=top_overlap(x1.values, x2.values, 5),
                                 dir_agree_pct=np.nan, n_levels=len(x1)))
            tops = {a: pdm[(pdm.outcome == o) & (pdm.dataset == ds) & (pdm.algorithm == a)].set_index("domain").auc_drop_x1000.idxmax()
                    for a in algs}
            rows.append(dict(analysis="Permutation (domain)", outcome=o, dataset=ds, pair="all three",
                             rho=np.nan, rho_lo=np.nan, rho_hi=np.nan, top3=np.nan, top5=np.nan,
                             dir_agree_pct=float(len(set(tops.values())) == 1) * 100, n_levels=len(DOM)))
            for a1, a2 in combinations(algs, 2):
                x1 = pdm[(pdm.outcome == o) & (pdm.dataset == ds) & (pdm.algorithm == a1)].set_index("domain").auc_drop_x1000
                x2 = pdm[(pdm.outcome == o) & (pdm.dataset == ds) & (pdm.algorithm == a2)].set_index("domain").auc_drop_x1000.reindex(x1.index)
                rows.append(dict(analysis="Permutation (domain)", outcome=o, dataset=ds, pair=f"{a1} vs {a2}",
                                 rho=spearmanr(x1, x2)[0], rho_lo=np.nan, rho_hi=np.nan, top3=np.nan, top5=np.nan,
                                 dir_agree_pct=np.nan, n_levels=len(x1)))
    return pv, pdm, pd.DataFrame(rows)


# ------------------------------------------------------------------ passo 4: domínio sozinho
def make_pre(cols):
    cat = [v for v in cols if v in c.CAT_FEAT]; num = [v for v in cols if v in c.NUM_FEAT]
    tr = []
    if cat:
        tr.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat))
    if num:
        tr.append(("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num))
    return ColumnTransformer(tr, remainder="drop")


def run_alone(data, models, algs=ALGS):
    rows = []
    for o in OUTC:
        Xtr, ytr = data[o]["train"]
        for a in algs:
            clf0 = models[(o, a)]["clf"]
            for dom, cols in {"All predictors": VARS, **DOM}.items():
                if dom == "All predictors":
                    pipe = models[(o, a)]                                  # modelo final, como treinado
                else:
                    pipe = Pipeline([("pre", make_pre(cols)), ("clf", clone(clf0))]).fit(Xtr[cols], ytr)
                for ds in VAL:
                    X, y = data[o][ds]; p = pipe.predict_proba(X if dom == "All predictors" else X[cols])[:, 1]
                    n = len(y); bs = [roc_auc_score(y[i], p[i]) for i in (rng.integers(0, n, n) for _ in range(td.B))]
                    rows.append(dict(outcome=o, algorithm=a, dataset=ds, domain=dom, auc=roc_auc_score(y, p),
                                     lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
            print(f"  [ALONE] {o} | {a} ok", flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ execução
def run(data, models, out_tab):
    out_tab.mkdir(parents=True, exist_ok=True)
    sv, sl, ss, cache = run_shap(data, models)
    pv, pdm, ps = run_perm(data, models, out_tab)
    al = run_alone(data, models)
    summ = pd.concat([ss, ps], ignore_index=True)
    for name, df in [("concordance_shap_variable", sv), ("concordance_shap_levels", sl), ("concordance_summary", summ),
                     ("concordance_perm_variable", pv), ("concordance_perm_domain", pdm), ("domain_alone", al)]:
        df.to_csv(out_tab / f"{name}.csv", index=False)
    with open(out_tab / "concordance_shap_cache.pkl", "wb") as f:
        pickle.dump(cache, f)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print("\n" + "=" * 100 + "\nRESUMO DA CONCORDÂNCIA\n" + "=" * 100)
    print(summ.round(3).to_string(index=False))
    print("\nTabelas salvas em", out_tab)


def main():
    import pyreadstat, joblib
    P = c.DEFAULT_PATHS
    el = {w: c.build_elsi(pyreadstat.read_dta(str(P[k]), encoding="latin1")[0], w)
          for w, k in [(1, "elsi1"), (2, "elsi2"), (3, "elsi3")]}
    sets = {"train": c.pool_elsi_train(el[1], el[2]), "ELSI W3": el[3],
            "PNS 2013": c.read_pns(P["pns2013"], 2013), "PNS 2019": c.read_pns(P["pns2019"], 2019)}
    data = {o: {k: c.get_Xy(d, t, a)[:2] for k, d in sets.items()} for o, (t, a, _) in OUTC.items()}
    root = c.ROOT / c.OUT_NAME
    models = {(o, a): joblib.load(root / "Models" / f"model_{stem}_{a}.joblib") for o, (_, _, stem) in OUTC.items() for a in ALGS}
    run(data, models, root / "Python_tables")
    print("\nConcluído.")


if __name__ == "__main__":
    main()
