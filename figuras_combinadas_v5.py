"""
figuras_combinadas_v5.py — figuras finais: XGBoost como modelo explicativo + concordância entre 3 algoritmos.
Letras dos painéis ficam NA BORDA do retângulo tracejado (canto superior esquerdo), com fundo branco que "corta" o tracejado.

Requer (mesma pasta): figuras_combinadas_v3.py, figuras_combinadas_v4.py (infraestrutura, Fig 1 e Fig 4).
Requer as tabelas de concordancia_modelos.py (Python_tables): concordance_shap_variable.csv, concordance_summary.csv,
concordance_perm_domain.csv, concordance_perm_variable.csv, domain_alone.csv, concordance_shap_cache.pkl

  Fig 1  Desenho + desempenho dos 3 modelos                                   (v4, letras na borda)
  Fig 2  SHAP e permutação por variável (XGBoost) | interações | concordância dos 3 algoritmos (SHAP, permutação)
  Fig 3  Domínios (XGBoost): permutação, contraste, share de SHAP, concordância entre algoritmos, resumo
  Fig 4  Barreiras + mapas + correlação ecológica (XGBoost)                    (v3, letras na borda)
  S1 barreiras por tercil | S2 permutação por variável | S3 OR plano x região | S4 SHAP RF e LogReg
  S5 domínio sozinho (3 algoritmos) | S6 como ler SHAP e permutação

Uso (em Codes/Python):  python figuras_combinadas_v5.py                 -> todas
                         python figuras_combinadas_v5.py 2 3 S4          -> só as indicadas
                         python figuras_combinadas_v5.py --demo          -> cache sintético (só Figs 1–2; teste de layout)
                         python figuras_combinadas_v5.py 1 --rebuild     -> refaz o cache (lê dados brutos)
"""
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter

import figuras_combinadas_v3 as v3
import figuras_combinadas_v4 as v4
from figuras_combinadas_v3 import (BLACK, C_PAP, C_MAM, EXAMS, DS3, MK_DS, GREY, LGREY, DGREY, DOM_ORDER, DOM_COL,
                                   DOM_NAME, TAB, new_fig, axmm, clean_y, VAR_LABEL, load_cache, MM)
from figuras_combinadas_v4 import canvas, box, arrow, T, marker_legend, SHORT, VARS8

EX_COL = dict(EXAMS)
ALGS = ["XGBoost", "RandomForest", "LogReg"]
ALG_LAB = {"XGBoost": "XGBoost", "RandomForest": "Random forest", "LogReg": "Logistic regression"}
ALG_SHORT = {"XGBoost": "XGB", "RandomForest": "RF", "LogReg": "LR"}
ALG_MK = {"XGBoost": ("D", True), "RandomForest": ("s", False), "LogReg": ("o", False)}

# ================================================================== LETRAS NA BORDA DO RETÂNGULO
_LET, _FR = {}, {}                                   # por figura: letras e retângulos registrados
_orig = dict(letter=v3.letter, title=v3.title, textmm=v3.textmm, frame=v3.frame, save=v3.save)


def _p_letter(fig, x, y, s):                         # código antigo: registra a letra (será desenhada na borda)
    _LET.setdefault(id(fig), []).append((x, y, s, True))


def _p_frame(fig, x, y, w, h, r=1.6):
    _FR.setdefault(id(fig), []).append((x, y, w, h))
    _orig["frame"](fig, x, y, w, h, r)


def _shift(fig, x):                                  # título/subtítulo que estava 6 mm à direita da letra volta para a borda
    for lx, ly, _, legacy in _LET.get(id(fig), []):
        if legacy and abs(x - (lx + 6)) < 1e-6:
            return lx
    return x


def _p_title(fig, x, y, s, fs=7.5):
    _orig["title"](fig, _shift(fig, x), y, s, fs)


def _p_textmm(fig, x, y, s, **kw):
    return _orig["textmm"](fig, _shift(fig, x), y, s, **kw)


def _p_save(fig, name):
    fr = _FR.get(id(fig), [])
    for lx, ly, s, legacy in _LET.get(id(fig), []):
        cands = [f for f in fr if f[0] - 0.01 <= lx + 1 <= f[0] + f[2] and f[1] - 0.01 <= ly - 2 <= f[1] + f[3]]
        if cands:                                    # letra "pendurada" no canto superior esquerdo do menor retângulo que a contém
            f = min(cands, key=lambda f_: f_[2] * f_[3])
            _orig["textmm"](fig, f[0] + 2.4, f[1] + 0.15, s, fontsize=10, fontweight="bold", ha="left", va="center", zorder=10,
                            bbox=dict(boxstyle="square,pad=0.18", fc="white", ec="none"))
        else:
            _orig["textmm"](fig, lx, ly, s, fontsize=11, fontweight="bold", ha="left", va="baseline")
    _orig["save"](fig, name)
    _LET.pop(id(fig), None); _FR.pop(id(fig), None)


for mod in (v3, v4):
    mod.letter, mod.title, mod.textmm, mod.frame, mod.save = _p_letter, _p_title, _p_textmm, _p_frame, _p_save
letter, title, textmm, save = _p_letter, _p_title, _p_textmm, _p_save


def tag(fig, x, y, w, h, let, r=1.6):
    """Retângulo tracejado + letra pendurada na borda (canto superior esquerdo)."""
    _p_frame(fig, x, y, w, h, r)
    _LET.setdefault(id(fig), []).append((x + 1, y + 4, let, False))


def ptitle(fig, x, y, ttl, sub=None, fs=7):
    """Título dentro do retângulo (x, y = canto superior esquerdo do retângulo)."""
    _orig["textmm"](fig, x + 2.5, y + 5.6, ttl, fontsize=fs, fontweight="bold", ha="left", va="baseline")
    if sub:
        _orig["textmm"](fig, x + 2.5, y + 9.0, sub, fontsize=6.0, ha="left", va="baseline")


def dumbbell(ax, y, exam, vals, ms=2.9, lw=0.9):
    """vals: dict algoritmo -> valor. Linha do mínimo ao máximo + marcador por algoritmo (cor = exame)."""
    col = EX_COL[exam]
    v = [vals[a] for a in ALGS if a in vals and np.isfinite(vals[a])]
    if len(v) > 1:
        ax.plot([min(v), max(v)], [y, y], color=col, lw=lw, zorder=2, solid_capstyle="butt")
    for a in ALGS:
        if a in vals and np.isfinite(vals[a]):
            mk, filled = ALG_MK[a]
            ax.plot(vals[a], y, mk, ms=ms, mfc=col if filled else "white", mec=col, mew=0.8, zorder=3, ls="none")


def model_legend(fig, x, y, step=15):
    marker_legend(fig, x, y, [(ALG_MK[a][0], BLACK if ALG_MK[a][1] else "white", ALG_SHORT[a]) for a in ALGS], step=step, fs=5.8)


def exam_words(fig, x_right, y):
    """Palavras 'Pap smear' / 'Mammography' coloridas, alinhadas à direita."""
    _orig["textmm"](fig, x_right, y, "Mammography", fontsize=5.8, color=C_MAM, ha="right", va="baseline", fontweight="bold")
    _orig["textmm"](fig, x_right - 17.5, y, "Pap smear", fontsize=5.8, color=C_PAP, ha="right", va="baseline", fontweight="bold")


def read(name):
    p = TAB / name
    if not p.exists():
        raise SystemExit(f"Falta {p.name}. Rode antes:  python concordancia_modelos.py")
    return pd.read_csv(p)



# ================================================================== FIGURA 1 (A4 em pé: design + ROC|calibração por amostra + AUC com IC | Brier)
def fig1(cache):
    from sklearn.metrics import roc_curve
    W, H = 180, 276
    fig = new_fig(W, H); cv = canvas(fig, W, H)
    # ---------- a) desenho do estudo
    tag(fig, 0.5, 1, 179, 63, "a"); ptitle(fig, 0.5, 1, "Study design", fs=8.5)
    y0 = 12; hb = 34.5
    box(cv, 3, y0, 44, hb)
    T(cv, 5, y0 + 4.2, "Training", 8, "bold"); T(cv, 5, y0 + 9.4, "ELSI waves 1 + 2", 7)
    T(cv, 5, y0 + 14.4, "Pap smear   n = 5,555", 7); T(cv, 5, y0 + 19.2, "Mammography   n = 7,258", 7)
    T(cv, 5, y0 + 24.8, "5-fold cross-validation", 7); T(cv, 5, y0 + 29.8, "Unweighted (weights: sensitivity)", 6.2)
    arrow(cv, 47, y0 + hb / 2, 56, y0 + hb / 2)
    box(cv, 56, y0, 46, hb, fc="#FFFFFF")
    T(cv, 58, y0 + 4.2, "Models (frozen)", 8, "bold"); T(cv, 58, y0 + 9.4, "XGBoost (primary)", 7)
    T(cv, 58, y0 + 14.0, "Logistic regression", 7); T(cv, 58, y0 + 18.6, "Random forest", 7)
    T(cv, 58, y0 + 25.0, "Explainability: SHAP,", 6.4); T(cv, 58, y0 + 29.4, "permutation, domains", 6.4)
    cv.plot([102, 107], [y0 + hb / 2, y0 + hb / 2], color=BLACK, lw=0.7, solid_capstyle="butt")
    vals = [("ELSI W3", "temporal validation", "n = 2,612 / 3,697"), ("PNS 2013", "external, retrospective", "n = 5,887 / 5,974"),
            ("PNS 2019", "external, prospective", "n = 9,862 / 11,310")]
    mids = []
    for k_, (nm, de, n_) in enumerate(vals):
        yb = y0 + k_ * 11.75
        box(cv, 111, yb, 66.5, 10.75, fc="#FFFFFF"); arrow(cv, 107, yb + 5.4, 111, yb + 5.4); mids.append(yb + 5.4)
        T(cv, 113, yb + 3.5, nm, 7.4, "bold"); T(cv, 113 + 17, yb + 3.5, de, 6.8)
        T(cv, 113, yb + 7.9, f"Pap / mammography   {n_}", 6.8)
    cv.plot([107, 107], [mids[0], mids[-1]], color=BLACK, lw=0.7, solid_capstyle="butt")
    ye = y0 + hb + 2.5
    box(cv, 111, ye, 66.5, 13)
    T(cv, 113, ye + 3.6, "Ecological analysis", 7.4, "bold"); T(cv, 113, ye + 7.6, "Painel-Oncologia · 2016–2024 · 25 states*", 6.4)
    T(cv, 113, ye + 10.8, "stage III–IV at diagnosis, by state (*AP, DF excluded)", 5.6)
    ym = ye + 6.5
    cv.plot([79, 79, 106.5], [y0 + hb, ym, ym], color=BLACK, lw=0.7, solid_capstyle="butt"); arrow(cv, 106.5, ym, 111, ym)
    T(cv, 82, ym - 2.6, "state-level predicted risk", 5.9); T(cv, 82, ym + 3.0, "(PNS 2019 women)", 5.9)

    # ---------- b–g) uma linha por amostra: ROC | calibração
    t2, ps = v4._perf_tables()
    heads = {"ELSI W3": "ELSI W3 (temporal)", "PNS 2013": "PNS 2013 (external)", "PNS 2019": "PNS 2019 (external)"}
    MSTY = {"XGBoost": ("-", 1.4, "D", True), "RandomForest": ((0, (4, 1.6)), 1.0, "s", False), "LogReg": ((0, (1, 1.3)), 1.0, "o", False)}
    pm = cache.get("pred_m") or {("XGBoost", e_, d_): v for (e_, d_), v in cache["pred"].items()}
    fxs = [0.5, 90.5]; fw = 89; Rs = [65.5, 118.5, 171.5]; fh = 52; ah, aw = 33, 54
    for r, ds in enumerate(DS3):
        for c_, kind in enumerate(["roc", "cal"]):
            tag(fig, fxs[c_], Rs[r], fw, fh, "bcdefg"[r * 2 + c_])
            ptitle(fig, fxs[c_], Rs[r], ("ROC · " if kind == "roc" else "Calibration · ") + heads[ds], fs=8)
            ax = axmm(fig, fxs[c_] + 17, Rs[r] + 11, aw, ah)
            ax.plot([0, 1], [0, 1], color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=0)
            for ex, col in EXAMS:
                for mname, (ls_, lw_, mk_, filled) in MSTY.items():
                    if (mname, ex, ds) not in pm:
                        continue
                    y, p = pm[(mname, ex, ds)]
                    if kind == "roc":
                        f, t, _ = roc_curve(y, p); ax.plot(f, t, color=col, lw=lw_, ls=ls_)
                    else:
                        mp, fp = v4.calib_points(y, p)
                        ax.plot(mp, fp, color=col, lw=lw_ * 0.8, ls=ls_, marker=mk_, ms=2.6, mfc=col if filled else "white", mew=0.6)
            if kind == "roc":                                         # AUC do XGBoost (IC 95%) dentro do painel
                for k_, (ex, col) in enumerate(EXAMS):
                    q = t2[(t2.Outcome == ex) & (t2.ds == ds)].iloc[0]
                    ax.text(0.97, 0.20 - k_ * 0.105, f"{'Pap' if ex == 'Pap smear' else 'Mammo'}  AUC {q.AUC:.2f} ({q.AUC_lo:.2f}–{q.AUC_hi:.2f})",
                            transform=ax.transAxes, ha="right", va="center", fontsize=6.6, color=col, fontweight="bold")
            ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
            ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"]); ax.set_yticklabels(["0", "0.5", "1"])
            ax.tick_params(labelsize=7.2)
            ax.set_xlabel("1 − specificity" if kind == "roc" else "Predicted probability", fontsize=7.6, labelpad=1.5)
            ax.set_ylabel("Sensitivity" if kind == "roc" else "Observed rate", fontsize=7.6, labelpad=1.5)

    # ---------- h) AUC por modelo (IC do XGBoost) | i) Brier (XGBoost)
    col_ps = {"ELSI W3": "AUC_W3", "PNS 2013": "AUC_PNS13", "PNS 2019": "AUC_PNS19"}
    pos = {(e_, d_): e_i * 3.7 + k_ for e_i, (e_, _) in enumerate(EXAMS) for k_, d_ in enumerate(DS3)}
    R4 = 224.5
    for panel, (fx_, let, ttl) in enumerate([(0.5, "h", "AUC, by model (XGBoost: 95% CI)"), (90.5, "i", "Brier score (XGBoost: 95% CI)")]):
        tag(fig, fx_, R4, 89, 44, let); ptitle(fig, fx_, R4, ttl, fs=8)
        ax = axmm(fig, fx_ + 27, R4 + 11, 56, 24)
        for (e_, d_), yv in pos.items():
            col = EX_COL[e_]
            if panel == 0:
                for mname, mk, off in [("LogReg", "o", -0.3), ("RandomForest", "s", 0.0), ("XGBoost", "D", 0.3)]:
                    v = ps[(ps.ex == e_) & (ps.Model == mname)][col_ps[d_]].iloc[0]
                    if mname == "XGBoost":
                        r_ = t2[(t2.Outcome == e_) & (t2.ds == d_)].iloc[0]; ax.plot([r_.AUC_lo, r_.AUC_hi], [yv + off] * 2, color=col, lw=1.0)
                    ax.plot(float(v), yv + off, mk, ms=3.0, mfc=col if mname == "XGBoost" else "white", mec=col, mew=0.8)
            else:
                r_ = t2[(t2.Outcome == e_) & (t2.ds == d_)].iloc[0]
                ax.plot([r_.Brier_lo, r_.Brier_hi], [yv] * 2, color=col, lw=1.0); ax.plot(r_.Brier, yv, "D", ms=3.0, mfc=col, mec=col)
        ax.set_ylim(max(pos.values()) + 0.7, -1.3); clean_y(ax)
        ax.set_yticks(list(pos.values())); ax.set_yticklabels([d_ for (_, d_) in pos], fontsize=6.8)
        ax.grid(axis="y", lw=0.3, color=LGREY); ax.set_axisbelow(True); ax.tick_params(axis="x", labelsize=7.2)
        for e_i, (e_, col) in enumerate(EXAMS):
            ax.text(0.0, e_i * 3.7 - 0.85, e_, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7, fontweight="bold")
        if panel == 0:
            ax.set_xlim(0.58, 0.70); ax.set_xticks([0.6, 0.65, 0.7]); ax.set_xlabel("AUC", fontsize=7.6, labelpad=1.5)
        else:
            ax.set_xlim(0.19, 0.26); ax.set_xticks([0.2, 0.22, 0.24, 0.26]); ax.set_xlabel("Brier score", fontsize=7.6, labelpad=1.5)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C_PAP, lw=0, label="Pap smear"), plt.Rectangle((0, 0), 1, 1, color=C_MAM, lw=0, label="Mammography"),
                        Line2D([0], [0], color=DGREY, lw=1.4, ls="-", marker="D", ms=3, mfc=DGREY, label="XGBoost (primary)"),
                        Line2D([0], [0], color=DGREY, lw=1.0, ls=(0, (4, 1.6)), marker="s", ms=3, mfc="white", label="Random forest"),
                        Line2D([0], [0], color=DGREY, lw=1.0, ls=(0, (1, 1.3)), marker="o", ms=3, mfc="white", label="Logistic regression")],
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5, frameon=False, handlelength=2.2, fontsize=7, columnspacing=1.2)
    save(fig, "fig1_performance")


# ================================================================== FIGURA 2
def fig2(cache):
    perm = pd.read_csv(TAB / "variable_importance_perm.csv"); perm = perm[perm.algorithm == "XGBoost"]
    order = perm.groupby("variable").auc_drop_x1000.mean().sort_values(ascending=False).index.tolist(); n = len(order)
    lev = read("concordance_shap_levels.csv"); csv = read("concordance_shap_variable.csv"); csp = read("concordance_perm_variable.csv"); summ = read("concordance_summary.csv")
    W, H = 180, 222
    fig = new_fig(W, H); cv = canvas(fig, W, H)
    sx, sw, px, pw, ah = 46, 54, 142, 36, 44
    feats, xl = v4.common_shap_setup(cache)
    rowY = [1.0, 79.5]
    for i, (ex, col) in enumerate(EXAMS):
        Y = rowY[i]; y0 = Y + 13
        tag(fig, 0.5, Y, 112, 77, "ab"[i]); tag(fig, 113.5, Y, 66, 77, "cd"[i])
        ax = axmm(fig, sx, y0, sw, ah)
        v4.shap_panel_common(ax, cache["shap"][ex], feats, xl); ax.set_xlabel("SHAP value (log-odds)")
        ptitle(fig, 0.5, Y, ex, f"SHAP · XGBoost, ELSI W3 (n = {cache['shap'][ex]['n']:,})")
        cax = axmm(fig, sx + sw + 5, y0 + ah / 2 - 15, 2.6, 30)
        cax.imshow(np.linspace(1, 0, 256)[:, None], aspect="auto", cmap=v3.SHAPMAP)
        cax.set_xticks([]); cax.set_yticks([0, 255]); cax.set_yticklabels(["High", "Low"], fontsize=6)
        cax.yaxis.tick_right(); cax.tick_params(length=0, pad=1.5)
        for s_ in cax.spines.values(): s_.set_visible(False)
        T(cv, sx + sw + 3.4, y0 + ah / 2, "Feature value", 5.8, ha="center", rotation=90)
        yb = y0 + ah + 13.0
        cv.annotate("", xy=(sx + 3, yb), xytext=(sx + sw / 2 - 2, yb), arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=0.6, mutation_scale=5))
        cv.annotate("", xy=(sx + sw - 3, yb), xytext=(sx + sw / 2 + 2, yb), arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=0.6, mutation_scale=5))
        T(cv, sx + 3, yb + 2.8, "lowers predicted risk", 5.8, ha="left"); T(cv, sx + sw - 3, yb + 2.8, "raises predicted risk", 5.8, ha="right")
        ax2 = axmm(fig, px, y0, pw, ah)
        for k, v in enumerate(order):
            for j, ds in enumerate(DS3):
                r = perm[(perm.outcome == ex) & (perm.dataset == ds) & (perm.variable == v)].iloc[0]; yv = k + (j - 1) * 0.24
                ax2.plot([r.lo, r.hi], [yv, yv], color=col, lw=0.8); ax2.plot(r.auc_drop_x1000, yv, MK_DS[ds], color=col, ms=2.6, ls="none")
        ax2.set_ylim(n - 0.4, -0.6); ax2.set_yticks(range(n)); ax2.set_yticklabels([VAR_LABEL[v] for v in order], fontsize=6.4); clean_y(ax2)
        ax2.grid(axis="y", lw=0.3, color=LGREY); ax2.set_axisbelow(True); ax2.axvline(0, color=DGREY, lw=0.5)
        ax2.set_xlabel(r"AUC drop ($\times10^{-3}$)")
        ptitle(fig, 113.5, Y, "Permutation importance", "XGBoost · 3 samples")
        if i == 0:
            ax2.legend(handles=[Line2D([0], [0], marker=MK_DS[d_], color=col, ms=3, lw=0.8, label=d_) for d_ in DS3],
                       loc="lower right", frameon=False, fontsize=5.8, handlelength=1.6, labelspacing=0.25, borderaxespad=0.1)

    # ---------------- linha 3: e) interações | f) concordância SHAP | g) concordância permutação
    Y3 = 159.5; y0 = Y3 + 13; ah3 = 36; hf = 66
    tag(fig, 0.5, Y3, 56, hf, "e"); tag(fig, 57.5, Y3, 61, hf, "f"); tag(fig, 119.5, Y3, 60, hf, "g")
    ia = pd.read_csv(TAB / "interaction_SHAP.csv")
    axe = axmm(fig, 25, y0, 29, ah3)
    y_ = 0; yt, yl = [], []
    for ex, col in EXAMS:
        top = ia[ia.outcome == ex].sort_values("mean_abs_interaction_x1000", ascending=False).head(3)
        for _, r in top.iterrows():
            a_, b_ = [p.strip() for p in r.pair.split(" x ")]
            axe.barh(y_, r.mean_abs_interaction_x1000, height=0.64, color=col, lw=0)
            yt.append(y_); yl.append(f"{SHORT[a_]} × {SHORT[b_]}"); y_ += 1
        y_ += 0.6
    axe.set_ylim(y_ - 0.6 + 0.4, -0.6); axe.set_yticks(yt); axe.set_yticklabels(yl, fontsize=6.1); clean_y(axe)
    axe.set_xlim(0, 52); axe.set_xticks([0, 25, 50]); axe.set_xlabel(r"|Interaction SHAP| ($\times10^{-3}$)", fontsize=6.4)
    ptitle(fig, 0.5, Y3, "Strongest interactions", "XGBoost · top 3 pairs per exam · W3")
    for k_, (ex, col) in enumerate(EXAMS):
        cv.add_patch(plt.Rectangle((3 + k_ * 20, y0 + ah3 + 11.4), 2.4, 2.0, color=col, lw=0)); T(cv, 6.3 + k_ * 20, y0 + ah3 + 12.4, ex, 5.4)

    ordc = csv[csv.algorithm == "XGBoost"].groupby("variable").share_pct.mean().sort_values(ascending=False).index.tolist()
    for (x_fr, w_fr, ttl, sub, let) in [(57.5, 61, "SHAP agreement", None, "f"), (119.5, 60, "Permutation agreement", None, "g")]:
        ax = axmm(fig, x_fr + 21.5, y0, w_fr - 24, ah3)
        for k, v in enumerate(ordc):
            for ex, off in [("Pap smear", -0.2), ("Mammography", 0.2)]:
                if let == "f":
                    vals = {a: float(csv[(csv.outcome == ex) & (csv.algorithm == a) & (csv.variable == v)].share_pct.iloc[0]) for a in ALGS}
                else:
                    vals = {a: float(csp[(csp.outcome == ex) & (csp.algorithm == a) & (csp.dataset == "ELSI W3") & (csp.variable == v)].auc_drop_x1000.iloc[0]) for a in ALGS}
                dumbbell(ax, k + off, ex, vals)
        ax.set_ylim(len(ordc) - 0.4, -0.6); ax.set_yticks(range(len(ordc))); ax.set_yticklabels([VAR_LABEL[v] for v in ordc], fontsize=6.2); clean_y(ax)
        ax.grid(axis="y", lw=0.3, color=LGREY); ax.set_axisbelow(True)
        if let == "f":
            ax.set_xlim(0, None); ax.set_xlabel("% of total |SHAP|", fontsize=6.6)
        else:
            ax.axvline(0, color=DGREY, lw=0.5); ax.set_xlabel(r"AUC drop ($\times10^{-3}$)", fontsize=6.6)
        ptitle(fig, x_fr, Y3, ttl)
        model_legend(fig, x_fr + 3.5, Y3 + 9.6, step=13)
        yy = y0 + ah3 + 12.4
        for k_, (ex, nm) in enumerate([("Pap smear", "Pap"), ("Mammography", "Mammo")]):
            if let == "f":
                s = summ[(summ.analysis == "SHAP (variable)") & (summ.outcome == ex) & (summ.pair != "all three")]
                w = lev[lev.outcome == ex].pivot_table(index=["variable", "level"], columns="algorithm", values="sign")
                nz = w[(w != 0).all(axis=1)]; k_same = int((nz.nunique(axis=1) == 1).sum())
                txt = f"{nm}: ρ {s.rho.min():.2f}–{s.rho.max():.2f} · same direction in {k_same}/{len(nz)} categories"
            else:
                s = summ[(summ.analysis == "Permutation (variable)") & (summ.outcome == ex)]
                txt = f"{nm}: ρ {s.rho.min():.2f}–{s.rho.max():.2f} over 3 model pairs × 3 samples"
            cv.add_patch(plt.Rectangle((x_fr + 2.5, yy + k_ * 3.0 - 1.0), 2.2, 2.0, color=EX_COL[ex], lw=0))
            T(cv, x_fr + 5.6, yy + k_ * 3.0, txt, 5.2)
    save(fig, "fig2_explainability")


# ================================================================== FIGURA 3
DOM_VARS = {"Access": "Region · Residence · Insurance", "Socioeconomic": "Education", "Social support": "Partner · Household size",
            "Controls": "Age group · Race/ethnicity"}
HYP = {"Access": ("H1", "Mammography > Pap"), "Socioeconomic": ("H2", "Pap > Mammography"), "Social support": ("H3", "Pap > Mammography")}


def fig3(_cache=None):
    imp = pd.read_csv(TAB / "domain_importance_block.csv"); imp = imp[(imp.algorithm == "XGBoost") & (imp.set == "principal")]
    con = pd.read_csv(TAB / "domain_contrast.csv"); con = con[(con.algorithm == "XGBoost") & (con.set == "principal")]
    shr = pd.read_csv(TAB / "domain_shap_share.csv")
    cpd = read("concordance_perm_domain.csv")
    ver = pd.read_csv(TAB / "hypothesis_verdict.csv").set_index("domain")
    W, H = 180, 148
    fig = new_fig(W, H); cv = canvas(fig, W, H)
    row = 5.2; gapd = 2.4; ty = 15.5
    pos, y_ = {}, 0.0
    for d in DOM_ORDER:
        for ds in DS3:
            pos[(d, ds)] = y_; y_ += 1
        y_ += gapd / row
    nrow_h = (y_ - gapd / row) * row + row
    ax_x = [56, 90, 122, 154]; ax_w = [26, 25, 25, 24]
    heads = [("a", "Importance", "XGBoost · permutation"), ("b", "Mammo − Pap", "XGBoost · 98.3% CI"),
             ("c", "Share of |SHAP|", "XGBoost · 95% CI"), ("d", "Other algorithms", "Δ AUC drop vs XGB")]
    fx = [44.5, 89.5, 121.5, 153.5]; fw = [44.5, 31.5, 31, 26]; Y = 1.5; fh = ty + nrow_h + 20 - Y
    axs = []
    for i in range(4):
        tag(fig, fx[i], Y, fw[i], fh, heads[i][0])
        ax = axmm(fig, ax_x[i], ty, ax_w[i], nrow_h)
        ax.set_ylim(max(pos.values()) + 0.6, -0.6); clean_y(ax); ax.set_axisbelow(True)
        ax.set_yticks(list(pos.values())); ax.set_yticklabels([ds if i == 0 else "" for (_, ds) in pos], fontsize=6)
        ax.patch.set_alpha(0)
        ptitle(fig, fx[i], Y, heads[i][1], heads[i][2], fs=6.8)
        axs.append(ax)
    for ex, col in EXAMS:
        off = -0.2 if ex == "Pap smear" else 0.2
        for (d, ds), yv in pos.items():
            r = imp[(imp.outcome == ex) & (imp.domain == d) & (imp.dataset == ds)].iloc[0]
            axs[0].plot([r.lo, r.hi], [yv + off] * 2, color=col, lw=0.8); axs[0].plot(r.auc_drop_x1000, yv + off, "o", color=col, ms=2.8)
            s = shr[(shr.outcome == ex) & (shr.dataset == ds) & (shr.domain == d)].iloc[0]
            axs[2].plot([s.lo, s.hi], [yv + off] * 2, color=col, lw=0.8); axs[2].plot(s.share_pct, yv + off, "o", color=col, ms=2.8)
            q = cpd[(cpd.outcome == ex) & (cpd.dataset == ds) & (cpd.domain == d)].set_index("algorithm").auc_drop_x1000   # d) diferença para o XGBoost
            for a in ("RandomForest", "LogReg"):
                mk, _ = ALG_MK[a]
                axs[3].plot(q[a] - q["XGBoost"], yv + off, mk, ms=2.8, mfc="white", mec=col, mew=0.8, ls="none")
    for (d, ds), yv in pos.items():
        r = con[(con.domain == d) & (con.dataset == ds)].iloc[0]
        axs[1].plot([r.lo98, r.hi98], [yv] * 2, color=BLACK, lw=0.8)
        axs[1].plot(r.contrast, yv, "D", ms=3, mfc=BLACK if r.excludes_zero else "white", mec=BLACK, mew=0.7)
    for ax in (axs[0], axs[1], axs[3]):
        ax.axvline(0, color=DGREY, lw=0.5)
    axs[0].set_xlabel(r"AUC drop ($\times10^{-3}$)"); axs[1].set_xlabel(r"Difference ($\times10^{-3}$)" + "\n← Pap     Mammo →")
    axs[2].set_xlabel("% of mean |SHAP|"); axs[2].set_xlim(0, None)
    axs[3].set_xlabel(r"Difference ($\times10^{-3}$)" + "\n← lower     higher →")
    xl = axs[1].get_xlim(); m = max(abs(xl[0]), abs(xl[1])); axs[1].set_xlim(-m, m)
    xl = axs[3].get_xlim(); m = max(abs(xl[0]), abs(xl[1])); axs[3].set_xlim(-m, m)
    # cartões de domínio (faixa colorida atravessa a–d)
    for d in DOM_ORDER:
        ys = [pos[(d, s_)] for s_ in DS3]

        def ymm(v): return ty + (v + 0.6) / (max(pos.values()) + 1.2) * nrow_h
        y0_, y1_ = ymm(min(ys) - 0.5), ymm(max(ys) + 0.5)
        cv.add_patch(plt.Rectangle((1, y0_), 177.5, y1_ - y0_, color=DOM_COL[d], alpha=0.13, lw=0))
        cv.add_patch(plt.Rectangle((1, y0_), 3.2, y1_ - y0_, color=DOM_COL[d], lw=0))
        T(cv, 6, y0_ + 3.2, DOM_NAME[d], 6.8, "bold"); T(cv, 6, y0_ + 7.0, DOM_VARS[d], 5.6)
        if d in HYP:
            h, exp = HYP[d]; vv = ver.loc[d, "verdict"]
            T(cv, 6, y0_ + 10.9, f"{h}: {exp}", 5.6)
            ok = vv == "SUPPORTED"
            box(cv, 6, y0_ + 12.0, 22 if ok else 25, 2.6, fc=BLACK if ok else "#FFFFFF", ec=BLACK, lw=0.5, r=0.8)
            T(cv, 6 + (11 if ok else 12.5), y0_ + 13.35, "Supported" if ok else "Not supported", 5.2, ha="center", **({"color": "white"} if ok else {}))
        else:
            T(cv, 6, y0_ + 10.9, "Adjustment covariates", 5.6)
    ly = ty + nrow_h + 15.6
    marker_legend(fig, 47, ly, [("o", C_PAP, "Pap smear"), ("o", C_MAM, "Mammography")], step=19, fs=5.8)
    marker_legend(fig, 90, ly, [("D", BLACK, "CI excl. 0"), ("D", "white", "CI incl. 0")], step=16, fs=5.6)
    marker_legend(fig, 157, ly, [("s", "white", "RF"), ("o", "white", "LR")], step=10, fs=5.6)
    # e) resumo — todas as frases saem das tabelas
    by = Y + fh + 3
    tag(fig, 0.5, by, 179, 33, "e")
    lines = [("Exploratory, pre-specified analysis; the 98.3% CI corrects for the three hypotheses.", True)]
    for d, (h, _) in HYP.items():
        r = ver.loc[d]
        lines.append((f"{h} {'supported' if r.verdict == 'SUPPORTED' else 'not supported'} ({DOM_NAME[d].split(' ', 1)[1]}): same sign in "
                      f"{'all 3 samples' if r.same_sign_3_samples else 'fewer than 3 samples'}; 98.3% CI excludes 0 in {int(r.ci_excludes_zero_n_of_3)}/3; "
                      f"logistic refit (Supplement) same sign in {int(r.test2_same_sign_n_of_3)}/3.", False))
    pv_ = cpd.pivot_table(index=["outcome", "dataset", "domain"], columns="algorithm", values="auc_drop_x1000")
    cc = []
    for ds in DS3:
        for d in DOM_ORDER:
            r = {"dataset": ds, "domain": d}
            for a_ in ALGS:
                r[a_] = pv_.loc[("Mammography", ds, d), a_] - pv_.loc[("Pap smear", ds, d), a_]
            cc.append(r)
    cc = pd.DataFrame(cc); cc["same_sign_3_algorithms"] = cc[ALGS].apply(lambda x: len(set(np.sign(x))) == 1, axis=1)
    cc.to_csv(TAB / "concordance_domain_contrast.csv", index=False)
    n_same = int(cc.same_sign_3_algorithms.sum()); bad = cc[~cc.same_sign_3_algorithms]
    top2 = 0; ntop = 0
    for (o_, ds_), g in cpd.groupby(["outcome", "dataset"]):
        t = {a_: set(h.nlargest(2, "auc_drop_x1000").domain) for a_, h in g.groupby("algorithm")}
        top2 += int(all(t[a_] == t["XGBoost"] for a_ in t)); ntop += 1
    exc = "; ".join(f"{r.domain} in {r.dataset}" for r in bad.itertuples()) or "none"
    lines.append((f"Mammo − Pap contrast: same sign in XGBoost, random forest and logistic regression in {n_same}/{len(cc)} domain × sample cells "
                  f"(exceptions: {exc}).", False))
    lines.append((f"The two leading domains were the same in the three algorithms in {top2}/{ntop} exam × sample comparisons.", False))
    for k, (s_, b_) in enumerate(lines):
        T(cv, 4, by + 7.6 + k * 4.4, s_, 5.9, "bold" if b_ else "normal")
    save(fig, "fig3_domains")


# ================================================================== SUPLEMENTO S4 — SHAP no RF e na regressão logística
def fig_s_shap():
    p = TAB / "concordance_shap_cache.pkl"
    if not p.exists():
        print("  S4 ignorada: falta concordance_shap_cache.pkl (rode concordancia_modelos.py)"); return
    with open(p, "rb") as f:
        sc = pickle.load(f)
    feats, _ = v4.common_shap_setup({"shap": {ex: sc[(ex, "XGBoost")] for ex, _ in EXAMS}})
    W, H = 180, 146
    fig = new_fig(W, H); cv = canvas(fig, W, H)
    fxs = [(0.5, 85), (85.5 + 0.5, 45), (131.5 + 0.5, 47.5)]
    axx = [40, 93, 140]; axw = [40, 37, 37]
    units = {"XGBoost": "SHAP value (log-odds)", "RandomForest": "SHAP value (probability)", "LogReg": "SHAP value (log-odds)"}
    letters = "abcdef"; ah = 44
    for r, (ex, col) in enumerate(EXAMS):
        Y = 1.0 + r * 68.5; y0 = Y + 13
        for c_, a in enumerate(ALGS):
            tag(fig, fxs[c_][0], Y, fxs[c_][1], 66, letters[r * 3 + c_])
            d = sc[(ex, a)]
            ax = axmm(fig, axx[c_], y0, axw[c_], ah)
            m = np.abs(d["shap"][:, [d["names"].index(f) for f in feats]]).max()
            v4.shap_panel_common(ax, d, feats, m * 1.05)
            ax.xaxis.set_major_locator(MaxNLocator(4, symmetric=True)); ax.xaxis.set_major_formatter(ScalarFormatter()); ax.set_xlabel(units[a], fontsize=6.6)
            if c_ > 0:
                ax.set_yticklabels([]); ax.tick_params(axis="y", length=0)
            ptitle(fig, fxs[c_][0], Y, f"{ex}", f"{ALG_LAB[a]} · ELSI W3 (n = {d['n']:,})")
    cax = axmm(fig, 70, H - 4.2, 40, 2.2)
    cax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=v3.SHAPMAP); cax.set_yticks([]); cax.set_xticks([0, 255])
    cax.set_xticklabels(["Low", "High"], fontsize=6); cax.tick_params(length=0, pad=1.5)
    for s_ in cax.spines.values(): s_.set_visible(False)
    T(cv, 90, H - 8.2, "Feature value", 5.8, ha="center")
    save(fig, "figS_SHAP_three_algorithms")


# ================================================================== SUPLEMENTO S5 — domínio sozinho
def fig_s_alone():
    al = read("domain_alone.csv")
    doms = ["All predictors"] + DOM_ORDER
    pos, y_ = {}, 0.0
    for d in doms:
        for ds in DS3:
            pos[(d, ds)] = y_; y_ += 1
        y_ += 0.9
    W, H = 180, 108
    fig = new_fig(W, H); cv = canvas(fig, W, H)
    xs = [(0.5, 89), (90.5, 89)]
    for i, (ex, col) in enumerate(EXAMS):
        Y = 1.0; tag(fig, xs[i][0], Y, xs[i][1], H - 9, "ab"[i])
        ax = axmm(fig, xs[i][0] + 24, Y + 13, 60, 74)
        for (d, ds), yv in pos.items():
            for a, off in zip(ALGS, (-0.25, 0.0, 0.25)):
                r = al[(al.outcome == ex) & (al.algorithm == a) & (al.domain == d) & (al.dataset == ds)].iloc[0]
                mk, filled = ALG_MK[a]
                ax.plot([r.lo, r.hi], [yv + off] * 2, color=col, lw=0.6)
                ax.plot(r.auc, yv + off, mk, ms=2.6, mfc=col if filled else "white", mec=col, mew=0.7, ls="none")
        ax.set_ylim(y_ - 0.9 + 0.3, -1.0); clean_y(ax)
        ax.set_yticks(list(pos.values())); ax.set_yticklabels([ds for (_, ds) in pos], fontsize=5.8)
        ax.axvline(0.5, color=DGREY, lw=0.5, ls=(0, (2, 2))); ax.set_xlim(0.44, 0.70); ax.set_xticks([0.5, 0.55, 0.6, 0.65, 0.7])
        ax.set_xticklabels(["0.50", "0.55", "0.60", "0.65", "0.70"]); ax.grid(axis="y", lw=0.3, color=LGREY); ax.set_axisbelow(True)
        ax.set_xlabel("AUC with the domain alone (0.50 = chance)", fontsize=6.6)
        for d in doms:
            y_first = pos[(d, DS3[0])]
            ax.text(1.0, y_first - 0.62, DOM_NAME.get(d, d), transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=6, fontweight="bold")
        ptitle(fig, xs[i][0], Y, ex, "model trained on ELSI W1+W2 with only the domain's predictors")
    model_legend(fig, 70, H - 4.2, step=26)
    save(fig, "figS_domain_alone")


# ================================================================== SUPLEMENTO S6 — como ler SHAP e permutação (arte)
def fig_s_howto():
    W, H = 180, 56
    fig = new_fig(W, H); cv = canvas(fig, W, H)
    tag(fig, 0.5, 1, 88, 53, "a"); tag(fig, 90.5, 1, 89, 53, "b")
    # a) SHAP
    ptitle(fig, 0.5, 1, "How to read SHAP", "schematic; not real data")
    fx = 6; xb = fx + 50; u = 1.9; cy = 13.5
    cv.plot([xb, xb], [cy + 6.8, cy + 26], color=DGREY, lw=0.6, ls=(0, (2, 1.5)))
    T(cv, xb - 1, cy + 3.8, "lowers ←", 5.4, ha="right"); T(cv, xb + 1, cy + 3.8, "→ raises", 5.4, ha="left")
    rng_f = np.random.default_rng(3)
    rows_f = [("No insurance", [(+5.5, 1), (+6.6, 1), (+7.5, 1), (+8.4, 1), (+6.0, 1), (+7.0, 1)], [(-1.6, 0), (-2.4, 0), (-3.0, 0), (-1.1, 0)]),
              ("Higher education", [(-4.0, 1), (-5.0, 1), (-5.8, 1), (-4.5, 1), (-6.4, 1)], [(+1.0, 0), (+1.8, 0), (+2.5, 0), (+1.4, 0), (+0.6, 0)]),
              ("Rural area", [(+2.0, 1), (+3.0, 1), (+3.8, 1), (+2.6, 1)], [(-1.0, 0), (-1.8, 0), (-2.5, 0), (-0.7, 0), (-1.4, 0)])]
    for k, (lab, hi_, lo_) in enumerate(rows_f):
        yk = cy + 10.2 + k * 5.6
        T(cv, fx, yk, lab, 5.6, ha="left")
        for xv, val in hi_ + lo_:
            cv.plot([xb + xv * u * 0.9], [yk + rng_f.uniform(-1.1, 1.1)], "o", ms=2.1, mew=0, color=C_MAM if val == 1 else C_PAP)
    gx0, gx1, gy = xb - 12, xb + 12, cy + 29.6
    cv.imshow(np.linspace(0, 1, 128)[None, :], extent=(gx0, gx1, gy + 1.3, gy - 1.3), aspect="auto", cmap=v3.SHAPMAP, zorder=3)
    cv.set_xlim(0, W); cv.set_ylim(H, 0)
    T(cv, gx0 - 1, gy, "no", 5.2, ha="right"); T(cv, gx1 + 1, gy, "yes", 5.2, ha="left")
    T(cv, xb, gy + 4.0, "colour = has the trait (feature value); each dot is one woman", 5.2, ha="center")
    # b) permutação
    ptitle(fig, 90.5, 1, "How to read permutation importance", "schematic; not real data")
    gx = 96; cell = 3.2; g0y = 19
    pat_a = [C_MAM, C_PAP, C_MAM, C_PAP]; pat_b = [C_PAP, C_PAP, C_MAM, C_MAM]
    for gi, (gx0_, pat, lab) in enumerate([(gx + 6, pat_a, "original data"), (gx + 52, pat_b, "one column shuffled")]):
        T(cv, gx0_ + 5.0, g0y - 4.0, lab, 5.6, ha="center")
        for c_ in range(3):
            for r_ in range(4):
                cv.add_patch(plt.Rectangle((gx0_ + c_ * (cell + 0.6), g0y + r_ * (cell + 0.6)), cell, cell, fc=(pat[r_] if c_ == 1 else "#D9D9D9"), ec="none"))
        cv.add_patch(plt.Rectangle((gx0_ + (cell + 0.6) - 0.5, g0y - 0.5), cell + 1.0, 4 * (cell + 0.6) - 0.1, fc="none", ec=BLACK, lw=0.6))
        T(cv, gx0_ + 5.0, g0y + 4 * (cell + 0.6) + 3.2, "AUC" if gi == 0 else "AUC ↓", 6.0, "bold", ha="center")
    arrow(cv, gx + 28, g0y + 7.5, gx + 50, g0y + 7.5); T(cv, gx + 39, g0y + 4.2, "shuffle", 5.2, ha="center")
    T(cv, 135, 47.5, "AUC drop = importance of that variable", 5.6, ha="center")
    save(fig, "figS_howto_SHAP_permutation")


# ================================================================== main
if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    todo = set(args) if args else {"1", "2", "3", "4", "S1", "S2", "S3", "S4", "S5", "S6"}
    cache = None
    if todo & {"1", "2"}:
        cache = load_cache(rebuild="--rebuild" in flags, demo="--demo" in flags)
    if "1" in todo: fig1(cache)
    if "2" in todo: fig2(cache)
    if "3" in todo: fig3()
    if "4" in todo: v3.fig4()
    if "S1" in todo: v3.fig_s_tertile()
    if "S2" in todo: v4.fig_s_perm()
    if "S3" in todo: v4.fig_s_or()
    if "S4" in todo: fig_s_shap()
    if "S5" in todo: fig_s_alone()
    if "S6" in todo: fig_s_howto()
    print("Fim.")
