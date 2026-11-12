"""
analise_dominios_v3.py — ANÁLISE EXPLORATÓRIA POR DOMÍNIO (pré-especificada em 22/10/2026)
Gera CSV, tabelas suplementares (Word) e figuras (PDF vetorial) em  <OUT_NAME>/Python_tables|Python_figures.
Mesma lógica e mesmas regras de decisão de teste_dominios.py (que só imprime).

DOMÍNIOS  Access (region, zone, insurance) | Socioeconomic (education) |
          Social support (marital, n_residents) | Controls (age_group, ethnicity)
HIPÓTESES (contraste = mamografia − Papanicolaou, queda de AUC x1000)
          H1 Access > 0 | H2 Socioeconomic < 0 | H3 Social support < 0
REGRA     Teste 1 (XGBoost): mesmo sinal nas 3 amostras e IC 98,3% exclui zero em >=2;
          Teste 2 (regressão logística refeita): mesmo sinal em >=2 das 3 amostras.
SENSIBILIDADE  plano de saúde movido de Access para Socioeconomic.

Execução (padrões = rodada de trabalho; para a rodada FINAL use mais reamostragens):
    python analise_dominios_v3.py                      (K_PERM=6, N_BOOT=500; ~20-30 min)
    $env:K_PERM="10"; $env:N_BOOT="1000"; python analise_dominios_v3.py     (final; ~1 h)
Saídas (todas em Outputs_v3_edu4):
  Python_tables/ domain_importance_block.csv, domain_contrast.csv, variable_importance_perm.csv,
                 domain_refit_LR.csv, domain_shap_share.csv, hypothesis_verdict.csv,
                 interaction_LRT.csv, interaction_OR.csv, interaction_AUC.csv, interaction_SHAP.csv,
                 barriers_pooled.csv, supplementary_tables_domains.docx
  Python_figures/ fig_domain_importance.pdf, fig_domain_contrast.pdf, fig_domain_shap_share.pdf,
                 fig_barriers_pooled.pdf, fig_interaction_OR.pdf
"""
import warnings; warnings.filterwarnings("ignore")
import os
os.environ["EDU_SCHEME"] = "4"; os.environ["OUT_NAME"] = "Outputs_v3_edu4"
os.environ.setdefault("K_PERM", "6"); os.environ.setdefault("N_BOOT", "500")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import roc_auc_score
import common as c
import teste_dominios as td
import teste_acesso_regiao_plano as ta

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
                     "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False})
COL = {"Pap smear": "#D55E00", "Mammography": "#0072B2"}
SAMPLE_COL = {"ELSI W3": "#7F7F7F", "PNS 2013": "#E69F00", "PNS 2019": "#009E73"}
VAL = ["ELSI W3", "PNS 2013", "PNS 2019"]

DOM = {"Access": ["region", "zone", "insurance"], "Socioeconomic": ["education"],
       "Social support": ["marital", "n_residents"], "Controls": ["age_group", "ethnicity"]}
DOM_S = {"Access excl. insurance": ["region", "zone"], "Socioeconomic + insurance": ["education", "insurance"]}
HYP = {"Access": ("H1", +1), "Socioeconomic": ("H2", -1), "Social support": ("H3", -1)}
VARS = c.ALL_FEAT
OUTC = {"Pap smear": ("target_pap", c.PAP_MAX_AGE, "pap_smear"),
        "Mammography": ("target_mamo", c.MAMO_MAX_AGE, "mammography")}
BAR_PAP = {1: "Never had sex", 2: "Not necessary", 3: "Embarrassment", 4: "Not oriented", 5: "Doesn't know where",
           6: "Financial difficulties", 7: "Long wait time", 8: "Distance/transport", 9: "Incompatible hours",
           10: "Insurance doesn't cover", 11: "Scheduled not done", 12: "Hysterectomy", 13: "Other", 14: "Other"}
BAR_MAMO = {1: "Scheduled not done", 2: "Not necessary", 3: "Didn't know where", 4: "Couldn't schedule",
            5: "Financial diff", 6: "Transport diff", 7: "Long wait time", 8: "Too far", 9: "Incompatible hours",
            10: "Insurance doesn't cover", 11: "Other"}


def wilson(k, n, z=1.96):
    p = k / n; ctr = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (ctr - h) * 100, (ctr + h) * 100


def fmt(m, lo, hi, d=1): return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def ci95(a): return np.percentile(a, [2.5, 97.5])


# ============================================================ cálculo
def compute(data, models_dir, algs):
    blocks = {**DOM, **DOM_S, **{"var:" + v: [v] for v in VARS}}
    import joblib
    t1 = {}
    for o, (_, _, stem) in OUTC.items():
        for alg in algs:
            pipe = joblib.load(models_dir / f"model_{stem}_{alg}.joblib")
            for ds in VAL:
                X, y = data[o][ds]; print(f"  [T1] {o} | {alg} | {ds} (n={len(y)})", flush=True)
                t1[(o, alg, ds)] = td.block_drops(pipe, X, y, blocks)
    rows_imp, rows_con, rows_var = [], [], []
    for (o, alg, ds), d in t1.items():
        for name, arr in d.items():
            lo, hi = ci95(arr * 1000)
            if name.startswith("var:"):
                rows_var.append(dict(outcome=o, algorithm=alg, dataset=ds, variable=name[4:],
                                     auc_drop_x1000=arr.mean() * 1000, lo=lo, hi=hi))
            else:
                rows_imp.append(dict(outcome=o, algorithm=alg, dataset=ds, domain=name,
                                     set="principal" if name in DOM else "sensitivity",
                                     auc_drop_x1000=arr.mean() * 1000, lo=lo, hi=hi))
    for alg in algs:
        for name in list(DOM) + list(DOM_S):
            for ds in VAL:
                dm, dp = t1[("Mammography", alg, ds)][name], t1[("Pap smear", alg, ds)][name]
                m, lo, hi = td.contrast(dm, dp)
                rows_con.append(dict(algorithm=alg, dataset=ds, domain=name,
                                     set="principal" if name in DOM else "sensitivity",
                                     mammography=dm.mean() * 1000, pap=dp.mean() * 1000, contrast=m,
                                     lo98=lo, hi98=hi, excludes_zero=bool(lo > 0 or hi < 0)))
    imp, con, var = pd.DataFrame(rows_imp), pd.DataFrame(rows_con), pd.DataFrame(rows_var)

    print("  [T2] regressão logística refeita (retirada de domínio)", flush=True)
    t2 = {o: td.drop_domain(data[o], DOM) for o in OUTC}
    rows = []
    for ds in VAL:
        for name in DOM:
            dm, am = t2["Mammography"][ds][name]; dp, ap = t2["Pap smear"][ds][name]
            m, lo, hi = td.contrast(dm, dp)
            rows.append(dict(dataset=ds, domain=name, drop_mammography=dm.mean() * 1000, drop_pap=dp.mean() * 1000,
                             contrast=m, lo98=lo, hi98=hi, auc_only_mammography=am, auc_only_pap=ap))
    t2df = pd.DataFrame(rows)

    print("  [T3] SHAP agrupado por domínio", flush=True)
    rows = []
    for o, (_, _, stem) in OUTC.items():
        pipe = joblib.load(models_dir / f"model_{stem}_XGBoost.joblib")
        for ds in VAL:
            X, y = data[o][ds]; base, sh = td.share_boot(td.shap_domain(pipe, X, DOM))
            for j, name in enumerate(DOM):
                lo, hi = ci95(sh[:, j])
                rows.append(dict(outcome=o, dataset=ds, domain=name, share_pct=base[j], lo=lo, hi=hi))
    shap_df = pd.DataFrame(rows)

    # veredito pré-especificado
    ver = []
    if "XGBoost" in algs:
        for name, (h, sgn) in HYP.items():
            s1 = con[(con.algorithm == "XGBoost") & (con.domain == name)].set_index("dataset").loc[VAL]
            same = bool(all(np.sign(s1.contrast) == sgn))
            ex = int(sum((s1.lo98 > 0) if sgn > 0 else (s1.hi98 < 0)))
            s2 = int(sum(np.sign(t2df[(t2df.domain == name)].set_index("dataset").loc[VAL].contrast) == sgn))
            ok = same and ex >= 2 and s2 >= 2
            ver.append(dict(hypothesis=h, domain=name, expected="Mammography > Pap" if sgn > 0 else "Pap > Mammography",
                            same_sign_3_samples=same, ci_excludes_zero_n_of_3=ex, test2_same_sign_n_of_3=s2,
                            verdict="SUPPORTED" if ok else "NOT supported"))
    return imp, con, var, t2df, shap_df, pd.DataFrame(ver)


def interaction(data, models_dir):
    import shap, joblib
    A, Bt, C, D = [], [], [], []
    for o, (_, _, stem) in OUTC.items():
        dd = data[o]
        for k, (X, y) in dd.items():
            for spec, nm in [("M1", "region x insurance (4 df)"), ("M2", "North/Northeast x insurance (1 df)")]:
                s, df, p = ta.lrt(X, y, spec)
                A.append(dict(outcome=o, dataset=k, interaction=nm, chi2=s, df=df, p=p))
            for lab, mask in [("North/Northeast", X["region"].isin(ta.LOW).values),
                              ("South/Southeast/Central-West", ~X["region"].isin(ta.LOW).values)]:
                if mask.sum() > 150 and len(set(y[mask])) == 2:
                    e, lo, hi = ta.or_no_insurance(X[mask].reset_index(drop=True), y[mask])
                    Bt.append(dict(outcome=o, dataset=k, stratum=lab, n=int(mask.sum()), OR=e, lo=lo, hi=hi))
        Xtr, ytr = dd["train"]; allX = {k: v[0] for k, v in dd.items()}
        for spec, nm in [("M0", "M0 main effects"), ("M1", "M1 + region x insurance"),
                         ("M2", "M2 + North/Northeast x insurance"), ("M3", "M3 composite access")]:
            Dm = ta.design(allX, spec); m = ta.fit(Dm["train"], ytr)
            r = dict(outcome=o, model=nm)
            for k in VAL: r[k] = roc_auc_score(dd[k][1], m.predict_proba(Dm[k])[:, 1])
            C.append(r)
        pipe = joblib.load(models_dir / f"model_{stem}_XGBoost.joblib"); X3 = dd["ELSI W3"][0]
        ohe = pipe["pre"].named_transformers_["cat"]["ohe"]
        names = ohe.get_feature_names_out(c.CAT_FEAT).tolist() + c.NUM_FEAT
        var = [next((v for v in c.CAT_FEAT if n.startswith(v + "_")), n) for n in names]
        I = shap.TreeExplainer(pipe["clf"]).shap_interaction_values(np.asarray(pipe["pre"].transform(X3)))
        vs = sorted(set(var)); idx = {v: [i for i, x in enumerate(var) if x == v] for v in vs}
        res = [(f"{vs[a]} x {vs[b]}", np.abs(I[:, idx[vs[a]], :][:, :, idx[vs[b]]].sum(axis=(1, 2)) * 2).mean() * 1000)
               for a in range(len(vs)) for b in range(a + 1, len(vs))]
        res = sorted(res, key=lambda x: -x[1])
        for rank, (k, v) in enumerate(res, 1):
            D.append(dict(outcome=o, pair=k, mean_abs_interaction_x1000=v, rank=rank))
    return pd.DataFrame(A), pd.DataFrame(Bt), pd.DataFrame(C), pd.DataFrame(D)


def barriers_pooled(pns):
    rows = []
    for ds, year in [("PNS 2013", 2013), ("PNS 2019", 2019)]:
        df = pns[ds]
        for o, (tgt, lim, _) in OUTC.items():
            if o == "Mammography" and year == 2019: continue
            col, lab = ("barrier_pap", BAR_PAP) if o == "Pap smear" else ("barrier_mamo", BAR_MAMO)
            s = df[(df["age"] >= c.AGE_MIN) & (df["age"] <= lim)].copy()
            s["age_group"] = pd.cut(s["age"], bins=[c.AGE_MIN - 1, c.AGE_CUT - 1, lim],
                                    labels=[f"{c.AGE_MIN}-{c.AGE_CUT-1}", f"{c.AGE_CUT}-{lim}"]).astype(str)
            s = s[(pd.to_numeric(s[tgt], errors="coerce") == 1) & pd.to_numeric(s[col], errors="coerce").notna()]
            s = s.dropna(subset=c.ALL_FEAT)
            lb = pd.to_numeric(s[col], errors="coerce").map(lab).dropna()
            n = len(lb)
            for k, v in lb.value_counts().items():
                lo, hi = wilson(v, n)
                rows.append(dict(dataset=ds, outcome=o, barrier=k, n_barrier=int(v), n=n, pct=v / n * 100, lo=lo, hi=hi))
    return pd.DataFrame(rows)


# ============================================================ figuras
def save(fig, path): fig.savefig(path, bbox_inches="tight"); plt.close(fig); print("  salvo:", path.name)


def fig_importance(imp, out):
    d = imp[(imp.algorithm == "XGBoost") & (imp.set == "principal")]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
    doms = list(DOM)
    for ax, ds in zip(axs, VAL):
        for j, o in enumerate(OUTC):
            s = d[(d.dataset == ds) & (d.outcome == o)].set_index("domain").loc[doms]
            y = np.arange(len(doms)) + (-0.18 if j == 0 else 0.18)
            ax.barh(y, s.auc_drop_x1000, height=0.34, color=COL[o], label=o if ds == VAL[0] else None)
            ax.errorbar(s.auc_drop_x1000, y, xerr=[s.auc_drop_x1000 - s.lo, s.hi - s.auc_drop_x1000],
                        fmt="none", ecolor="black", elinewidth=0.8, capsize=1.8)
        ax.set_title(ds); ax.set_yticks(range(len(doms))); ax.set_yticklabels(doms)
        ax.set_xlabel("AUC drop ×1000"); ax.axvline(0, color="black", lw=0.6)
    axs[0].invert_yaxis()
    fig.legend(loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(); save(fig, out / "fig_domain_importance.pdf")


def fig_contrast(con, ver, out):
    d = con[(con.algorithm == "XGBoost") & (con.set == "principal")]
    doms = list(DOM); fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for j, ds in enumerate(VAL):
        for i, name in enumerate(doms):
            r = d[(d.dataset == ds) & (d.domain == name)].iloc[0]; y = i + (j - 1) * 0.22
            ax.errorbar(r.contrast, y, xerr=[[r.contrast - r.lo98], [r.hi98 - r.contrast]], fmt="o", ms=4,
                        color=SAMPLE_COL[ds], ecolor=SAMPLE_COL[ds], elinewidth=1.1, capsize=2,
                        label=ds if i == 0 else None)
    ax.axvline(0, color="black", lw=0.7)
    labs = []
    for name in doms:
        v = ver[ver.domain == name]
        labs.append(f"{name}\n({v.hypothesis.iloc[0]}: {v.verdict.iloc[0].lower()})" if len(v) else name)
    ax.set_yticks(range(len(doms))); ax.set_yticklabels(labs); ax.invert_yaxis()
    ax.set_xlabel("Contrast in AUC drop ×1000 (mammography − Pap smear), 98.3% CI")
    ax.text(0.01, 1.03, "← larger for Pap smear", transform=ax.transAxes, fontsize=8, ha="left")
    ax.text(0.99, 1.03, "larger for mammography →", transform=ax.transAxes, fontsize=8, ha="right")
    ax.legend(frameon=False, loc="lower right"); fig.tight_layout(); save(fig, out / "fig_domain_contrast.pdf")


def fig_shap(shap_df, out):
    fig, ax = plt.subplots(figsize=(6.0, 3.0)); pal = ["#4C72B0", "#DD8452", "#55A868", "#B0B0B0"]
    labels, ypos = [], 0
    for ds in VAL:
        for o in OUTC:
            s = shap_df[(shap_df.dataset == ds) & (shap_df.outcome == o)].set_index("domain").loc[list(DOM)]
            left = 0
            for k, name in enumerate(DOM):
                ax.barh(ypos, s.share_pct[name], left=left, color=pal[k], edgecolor="white", height=0.7,
                        label=name if ypos == 0 else None)
                if s.share_pct[name] > 8: ax.text(left + s.share_pct[name] / 2, ypos, f"{s.share_pct[name]:.0f}",
                                                  ha="center", va="center", fontsize=7.5, color="white" if k < 3 else "black")
                left += s.share_pct[name]
            labels.append(f"{ds} · {'Pap' if o == 'Pap smear' else 'Mammo'}"); ypos += 1
        ypos += 0.4
    ys = [i + 0.4 * (i // 2) for i in range(len(labels))]
    ax.set_yticks(ys); ax.set_yticklabels(labels); ax.invert_yaxis(); ax.set_xlabel("Share of mean |SHAP| (%)")
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.18)); fig.tight_layout()
    save(fig, out / "fig_domain_shap_share.pdf")


def fig_barriers(bp, out):
    panels = [("PNS 2013", "Pap smear", "A  Pap smear — PNS 2013"), ("PNS 2013", "Mammography", "B  Mammography — PNS 2013"),
              ("PNS 2019", "Pap smear", "C  Pap smear — PNS 2019")]
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.3))
    for ax, (ds, o, title) in zip(axs, panels):
        s = bp[(bp.dataset == ds) & (bp.outcome == o)].sort_values("pct", ascending=False).head(7)
        y = np.arange(len(s))
        ax.barh(y, s.pct, color=COL[o], height=0.65)
        ax.errorbar(s.pct, y, xerr=[s.pct - s.lo, s.hi - s.pct], fmt="none", ecolor="black", elinewidth=0.8, capsize=1.8)
        ax.set_yticks(y); ax.set_yticklabels(s.barrier); ax.invert_yaxis()
        ax.set_title(f"{title}\n(n={int(s.n.iloc[0]):,})", fontsize=8.5, loc="left"); ax.set_xlabel("% of non-adherent women")
    fig.tight_layout(); save(fig, out / "fig_barriers_pooled.pdf")


def fig_or(orr, out):
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.8), sharex=True)
    for ax, o in zip(axs, OUTC):
        d = orr[orr.outcome == o]; k = 0
        for ds in ["ELSI W1+W2 (train)", "ELSI W3", "PNS 2013", "PNS 2019"]:
            key = "train" if ds.startswith("ELSI W1") else ds
            for j, g in enumerate(["North/Northeast", "South/Southeast/Central-West"]):
                r = d[(d.dataset == key) & (d.stratum == g)]
                if len(r) == 0: continue
                r = r.iloc[0]; y = k + (j - 0.5) * 0.3
                ax.errorbar(r.OR, y, xerr=[[r.OR - r.lo], [r.hi - r.OR]], fmt="o", ms=4, capsize=2, elinewidth=1,
                            color="#C44E52" if j == 0 else "#4C72B0", label=g if k == 0 else None)
            k += 1
        ax.axvline(1, color="black", lw=0.6); ax.set_yticks(range(4))
        ax.set_yticklabels(["ELSI W1+W2", "ELSI W3", "PNS 2013", "PNS 2019"]); ax.invert_yaxis()
        ax.set_title(o); ax.set_xlabel("Adjusted OR of no private insurance")
    axs[0].legend(frameon=False, loc="lower right", fontsize=7.5); fig.tight_layout(); save(fig, out / "fig_interaction_OR.pdf")


# ============================================================ Word
def make_docx(path, imp, con, var, t2df, shap_df, ver, A, Bt, C, bp, algs):
    from docx import Document
    from docx.shared import Pt
    from docx.enum.section import WD_ORIENT
    doc = Document(); sec = doc.sections[0]; sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width, sec.page_height = sec.page_height, sec.page_width
    doc.styles["Normal"].font.size = Pt(8)

    def add(title, df, note=""):
        doc.add_paragraph().add_run(title).bold = True
        t = doc.add_table(rows=1, cols=len(df.columns)); t.style = "Light Grid Accent 1"
        for i, h in enumerate(df.columns): t.rows[0].cells[i].text = str(h)
        for _, r in df.iterrows():
            cells = t.add_row().cells
            for i, v in enumerate(r): cells[i].text = str(v)
        if note: doc.add_paragraph(note).runs[0].font.size = Pt(7.5)
        doc.add_page_break()

    pr = con[con.set == "principal"]
    rows = [dict(Algorithm=r.algorithm, Domain=r.domain, Sample=r.dataset, **{
        "Pap smear": f"{r.pap:.1f}", "Mammography": f"{r.mammography:.1f}",
        "Contrast [98.3% CI]": fmt(r.contrast, r.lo98, r.hi98), "CI excludes 0": "yes" if r.excludes_zero else "no"})
        for r in pr.itertuples()]
    add("Table S-A1. Domain-level block-permutation importance (AUC drop ×1000) and contrast between exams",
        pd.DataFrame(rows), "Contrast = mammography − Pap smear. Domains: Access (region, zone, insurance); Socioeconomic "
        "(education); Social support (marital status, household size); Controls (age group, ethnicity). Bootstrap CIs over women.")
    sn = con[(con.set == "sensitivity") & (con.algorithm == "XGBoost")]
    add("Table S-A2. Sensitivity: insurance moved from Access to Socioeconomic (XGBoost)",
        pd.DataFrame([dict(Domain=r.domain, Sample=r.dataset, **{"Pap smear": f"{r.pap:.1f}",
                      "Mammography": f"{r.mammography:.1f}", "Contrast [98.3% CI]": fmt(r.contrast, r.lo98, r.hi98)})
                      for r in sn.itertuples()]))
    add("Table S-A3. Logistic regression refit on ELSI W1+W2: AUC drop (×1000) when the domain is removed, and AUC of the domain alone",
        pd.DataFrame([dict(Sample=r.dataset, Domain=r.domain, **{"Drop, Pap": f"{r.drop_pap:.1f}", "Drop, Mammo": f"{r.drop_mammography:.1f}",
                      "Contrast [98.3% CI]": fmt(r.contrast, r.lo98, r.hi98), "AUC alone, Pap": f"{r.auc_only_pap:.3f}",
                      "AUC alone, Mammo": f"{r.auc_only_mammography:.3f}"}) for r in t2df.itertuples()]))
    add("Table S-A4. Grouped SHAP (XGBoost): share of mean |SHAP| by domain",
        pd.DataFrame([dict(Sample=r.dataset, Outcome=r.outcome, Domain=r.domain, **{"Share % [95% CI]": fmt(r.share_pct, r.lo, r.hi)})
                      for r in shap_df.itertuples()]))
    if len(ver):
        add("Table S-A5. Pre-specified hypotheses (rule defined before the domain analyses)", ver.rename(columns=str))
    v = var.copy(); v["val"] = [fmt(a, b, c_) for a, b, c_ in zip(v.auc_drop_x1000, v.lo, v.hi)]
    add("Table S-B. Permutation importance by variable (AUC drop ×1000 [95% CI])",
        v.pivot_table(index=["algorithm", "dataset", "variable"], columns="outcome", values="val", aggfunc="first").reset_index())
    add("Table S-C1. Likelihood-ratio tests of region × insurance interaction",
        A.assign(chi2=A.chi2.round(2), p=A.p.map(lambda x: f"{x:.4f}")))
    add("Table S-C2. Adjusted OR of no private insurance by region group",
        Bt.assign(OR=[fmt(a, b, c_, 2) for a, b, c_ in zip(Bt.OR, Bt.lo, Bt.hi)])[["outcome", "dataset", "stratum", "n", "OR"]])
    add("Table S-C3. Out-of-sample AUC (logistic regression trained on ELSI W1+W2)", C.round(3))
    add("Table S-D. Pooled self-reported barriers among non-adherent women (PNS)",
        bp.assign(**{"% [95% CI]": [fmt(a, b, c_) for a, b, c_ in zip(bp.pct, bp.lo, bp.hi)]})[["dataset", "outcome", "barrier", "n_barrier", "n", "% [95% CI]"]])
    doc.save(path); print("  salvo:", path.name)


# ============================================================ execução
def run(data, pns, models_dir, out_tab, out_fig, algs):
    out_tab.mkdir(parents=True, exist_ok=True); out_fig.mkdir(parents=True, exist_ok=True)
    imp, con, var, t2df, shap_df, ver = compute(data, models_dir, algs)
    A, Bt, C, D = interaction(data, models_dir)
    bp = barriers_pooled(pns)
    for name, df in [("domain_importance_block", imp), ("domain_contrast", con), ("variable_importance_perm", var),
                     ("domain_refit_LR", t2df), ("domain_shap_share", shap_df), ("hypothesis_verdict", ver),
                     ("interaction_LRT", A), ("interaction_OR", Bt), ("interaction_AUC", C), ("interaction_SHAP", D),
                     ("barriers_pooled", bp)]:
        df.to_csv(out_tab / f"{name}.csv", index=False)
    print("  CSV salvos em", out_tab)
    make_docx(out_tab / "supplementary_tables_domains.docx", imp, con, var, t2df, shap_df, ver, A, Bt, C, bp, algs)
    if "XGBoost" in algs:
        fig_importance(imp, out_fig); fig_contrast(con, ver, out_fig)
    fig_shap(shap_df, out_fig); fig_barriers(bp, out_fig); fig_or(Bt, out_fig)
    print("\nVEREDITO PRÉ-ESPECIFICADO:"); print(ver.to_string(index=False) if len(ver) else "(XGBoost ausente)")


def main():
    import pyreadstat
    P = c.DEFAULT_PATHS
    el = {w: c.build_elsi(pyreadstat.read_dta(str(P[k]), encoding="latin1")[0], w)
          for w, k in [(1, "elsi1"), (2, "elsi2"), (3, "elsi3")]}
    pns = {"PNS 2013": c.read_pns(P["pns2013"], 2013), "PNS 2019": c.read_pns(P["pns2019"], 2019)}
    sets = {"train": c.pool_elsi_train(el[1], el[2]), "ELSI W3": el[3], **pns}
    data = {o: {k: c.get_Xy(d, t, a)[:2] for k, d in sets.items()} for o, (t, a, _) in OUTC.items()}
    algs = os.environ.get("ALGS", "XGBoost,LogReg").split(",")
    root = c.ROOT / c.OUT_NAME
    run(data, pns, root / "Models", root / "Python_tables", root / "Python_figures", algs)
    print("\nConcluído.")


if __name__ == "__main__":
    main()
