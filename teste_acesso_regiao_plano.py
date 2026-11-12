"""
TESTE EXPLORATÓRIO — "mecanismo de acesso": o plano de saúde pesa mais onde o SUS tem menos oferta?
Não grava nada em disco: só imprime. Rode no VS Code a partir de Codes/Python.

Hipótese (definida pela teoria, antes de ver resultados): o efeito de NÃO ter plano de saúde sobre a
não adesão é MAIOR em regiões com menor oferta pública de exames (Norte e Nordeste) do que no
Sul/Sudeste/Centro-Oeste  (modificação de efeito região x plano).

A  Teste formal de interação (razão de verossimilhança), regressão logística, em cada amostra:
     M0  efeitos principais (8 preditores)
     M1  M0 + região x plano (4 gl)              -> interação completa
     M2  M0 + grupo de região x plano (1 gl)     -> interação pela hipótese (N/NE vs demais)
B  OR de NÃO ter plano, ajustado, dentro de cada região e de cada grupo (IC 95% por bootstrap).
C  Desempenho fora da amostra (treino ELSI W1+W2 -> W3, PNS 2013, PNS 2019):
     M0 vs M1 vs M2 vs M3 (variável composta "acesso" = grupo de região x plano, no lugar das duas).
D  XGBoost salvo (Outputs_v3_edu4): interação SHAP entre pares de variáveis em ELSI W3
     (região x plano comparada a todos os outros pares).
"""
import warnings; warnings.filterwarnings("ignore")
import os
os.environ["EDU_SCHEME"] = "4"; os.environ["OUT_NAME"] = "Outputs_v3_edu4"
import numpy as np, pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
import common as c

B_BOOT = int(os.environ.get("N_BOOT", 200))
LOW = {"North", "Northeast"}                     # menor oferta pública (hipótese a priori)
rng = np.random.default_rng(42)
CAT_OTHER = ["age_group", "zone", "marital", "ethnicity", "education"]


# ----------------------------------------------------------------------- desenho
def design(frames, spec):
    """frames: dict nome -> X (DataFrame com ALL_FEAT). Devolve dict nome -> matriz numérica,
    com colunas idênticas entre amostras (níveis fixos)."""
    big = pd.concat([f.assign(_k=k) for k, f in frames.items()], ignore_index=True)
    big["reg_grp"] = np.where(big["region"].isin(LOW), "NorthNE", "Other")
    big["ins_no"] = (big["insurance"] == "No").astype(float)
    if spec == "M3":                                           # composto: grupo x plano
        big["access"] = big["reg_grp"] + "_" + big["insurance"]
        cats = CAT_OTHER + ["access"]
    else:
        cats = CAT_OTHER + ["region", "insurance"]
    Z = pd.get_dummies(big[cats], drop_first=True).astype(float)
    Z["n_residents"] = big["n_residents"].astype(float)
    if spec == "M1":
        for r in sorted(set(big["region"])):
            if r != "South":
                Z[f"ins_no_x_{r}"] = big["ins_no"] * (big["region"] == r)
    if spec == "M2":
        Z["ins_no_x_lowsupply"] = big["ins_no"] * (big["reg_grp"] == "NorthNE")
    Z = (Z - Z.mean()) / Z.std().replace(0, 1)                 # padroniza (estabilidade numérica)
    return {k: Z[big["_k"] == k].reset_index(drop=True) for k in frames}


def fit(Z, y):
    return LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000).fit(Z, y)


def loglik(m, Z, y):
    return -log_loss(y, m.predict_proba(Z)[:, 1], normalize=False)


def lrt(X, y, spec):
    d0 = design({"d": X}, "M0")["d"]; d1 = design({"d": X}, spec)["d"]
    ll0 = loglik(fit(d0, y), d0, y); ll1 = loglik(fit(d1, y), d1, y)
    df = d1.shape[1] - d0.shape[1]
    return 2 * (ll1 - ll0), df, chi2.sf(2 * (ll1 - ll0), df)


def or_no_insurance(X, y):
    """OR ajustado de 'sem plano' (demais covariáveis, sem região). IC bootstrap."""
    def one(Xs, ys):
        Z = pd.get_dummies(Xs[CAT_OTHER + ["insurance"]], drop_first=True).astype(float)
        Z["n_residents"] = Xs["n_residents"].astype(float).values
        m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000).fit(Z, ys)
        return np.exp(m.coef_[0][list(Z.columns).index("insurance_Yes")] * -1)   # No vs Yes
    est = one(X, y); n = len(y); bs = []
    for _ in range(B_BOOT):
        i = rng.integers(0, n, n)
        if len(set(y[i])) < 2 or X.iloc[i]["insurance"].nunique() < 2: continue
        try: bs.append(one(X.iloc[i], y[i]))
        except Exception: pass
    lo, hi = np.percentile(bs, [2.5, 97.5]); return est, lo, hi


# ----------------------------------------------------------------------- dados
def main():
    import pyreadstat, joblib
    P = c.DEFAULT_PATHS
    el = {w: c.build_elsi(pyreadstat.read_dta(str(P[k]), encoding="latin1")[0], w)
          for w, k in [(1, "elsi1"), (2, "elsi2"), (3, "elsi3")]}
    train = c.pool_elsi_train(el[1], el[2])
    sets = {"ELSI W1+W2 (treino)": train, "ELSI W3": el[3],
            "PNS 2013": c.read_pns(P["pns2013"], 2013), "PNS 2019": c.read_pns(P["pns2019"], 2019)}
    OUT = {"Pap smear": ("target_pap", c.PAP_MAX_AGE, "pap_smear"),
           "Mammography": ("target_mamo", c.MAMO_MAX_AGE, "mammography")}
    pd.set_option("display.width", 170)

    for outc, (tgt, amax, stem) in OUT.items():
        data = {k: c.get_Xy(d, tgt, amax)[:2] for k, d in sets.items()}
        print("\n" + "#" * 100 + f"\n{outc.upper()}\n" + "#" * 100)

        print("\nA) Teste de interação (razão de verossimilhança)")
        rows = []
        for k, (X, y) in data.items():
            for spec, nome in [("M1", "região x plano (4 gl)"), ("M2", "N/NE x plano (1 gl)")]:
                s, df, p = lrt(X, y, spec)
                rows.append(dict(amostra=k, interacao=nome, qui2=round(s, 2), gl=df, p=f"{p:.4f}"))
        print(pd.DataFrame(rows).to_string(index=False))

        print("\nB) OR ajustado de NÃO ter plano de saúde [IC 95% bootstrap], por região e grupo")
        rows = []
        for k, (X, y) in data.items():
            for lab, mask in [("Norte/Nordeste", X["region"].isin(LOW).values),
                              ("Sul/Sudeste/Centro-Oeste", ~X["region"].isin(LOW).values)]:
                if mask.sum() > 150 and len(set(y[mask])) == 2:
                    e, lo, hi = or_no_insurance(X[mask].reset_index(drop=True), y[mask])
                    rows.append(dict(amostra=k, estrato=lab, n=int(mask.sum()),
                                     OR=f"{e:.2f} [{lo:.2f}, {hi:.2f}]"))
        print(pd.DataFrame(rows).to_string(index=False))

        print("\nC) Desempenho fora da amostra — AUC (treino ELSI W1+W2, regressão logística sem penalização)")
        Xtr, ytr = data["ELSI W1+W2 (treino)"]
        rows = []
        for spec in ["M0", "M1", "M2", "M3"]:
            allX = {k: v[0] for k, v in data.items()}
            D = design(allX, spec); m = fit(D["ELSI W1+W2 (treino)"], ytr)
            r = {"modelo": {"M0": "M0 efeitos principais", "M1": "M1 + região x plano",
                            "M2": "M2 + N/NE x plano", "M3": "M3 composto 'acesso'"}[spec]}
            for k in ["ELSI W3", "PNS 2013", "PNS 2019"]:
                r[k] = round(roc_auc_score(data[k][1], m.predict_proba(D[k])[:, 1]), 3)
            rows.append(r)
        t = pd.DataFrame(rows); print(t.to_string(index=False))

        print("\nD) Interação SHAP (XGBoost salvo), ELSI W3 — média |interação| entre pares de variáveis (x1000)")
        import shap
        pipe = joblib.load(c.ROOT / c.OUT_NAME / "Models" / f"model_{stem}_XGBoost.joblib")
        X3, y3 = data["ELSI W3"]
        ohe = pipe["pre"].named_transformers_["cat"]["ohe"]
        names = ohe.get_feature_names_out(c.CAT_FEAT).tolist() + c.NUM_FEAT
        var = [next((v for v in c.CAT_FEAT if n.startswith(v + "_")), n) for n in names]
        M = np.asarray(pipe["pre"].transform(X3))
        I = shap.TreeExplainer(pipe["clf"]).shap_interaction_values(M)        # (n, f, f)
        vs = sorted(set(var)); idx = {v: [i for i, x in enumerate(var) if x == v] for v in vs}
        res = []
        for a in range(len(vs)):
            for b in range(a + 1, len(vs)):
                val = np.abs(I[:, idx[vs[a]], :][:, :, idx[vs[b]]].sum(axis=(1, 2)) * 2).mean() * 1000
                res.append((f"{vs[a]} x {vs[b]}", val))
        res = sorted(res, key=lambda x: -x[1])
        ri = [v for k, v in res if k in ("insurance x region", "region x insurance")][0]
        print("  Top 6 pares:"); [print(f"    {k:28s}{v:6.1f}") for k, v in res[:6]]
        print(f"  região x plano: {ri:.1f}  (posição {[k for k, _ in res].index('insurance x region') + 1} de {len(res)})")
    print("\nFim. Nada foi salvo em disco.")


if __name__ == "__main__":
    main()
