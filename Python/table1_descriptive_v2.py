"""
Table 1 — Descriptive statistics
New design: ELSI W1, W2, W3 (train+temporal) | PNS 2013, PNS 2019 (external)
Output: Outputs/Python_tables/table1_descriptive.docx + .csv
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pyreadstat
from pathlib import Path
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ==============================================================================
# PATHS
# ==============================================================================
ROOT    = Path(__file__).parent.parent.parent
DATA    = ROOT / "Data"
OUT_TAB = ROOT / "Outputs" / "Python_tables"
OUT_TAB.mkdir(parents=True, exist_ok=True)

ELSI_W1 = DATA / "ELSI Portugues (1a onda) stata13.dta"
ELSI_W2 = DATA / "ELSI Portugues (2a onda) stata13.dta"
ELSI_W3 = DATA / "ELSI Portugues (3a onda).dta"
PNS13   = DATA / "PNS_2013.csv"
PNS19   = DATA / "pns2019.csv"

# ==============================================================================
# CONSTANTS — identical to pns_pipeline_v2.py
# ==============================================================================
MISS = {9,99,999,88,8}
def clean(s): return pd.to_numeric(s, errors="coerce").replace(list(MISS), np.nan)

ETHNIC_ELSI = {1:"White",2:"Black",3:"Other",4:"Other",5:"Other"}
ETHNIC_PNS  = {1:"White",2:"Black",3:"Other",4:"Brown", 5:"Other"}

# Brown separate for PNS display
ETHNIC_ELSI_FULL = {1:"White",2:"Black",3:"Brown",4:"Other",5:"Other"}
ETHNIC_PNS_FULL  = {1:"White",2:"Black",3:"Other",4:"Brown", 5:"Other"}

MAR_ELSI  = {1:"Without partner",2:"With partner",
             3:"Without partner",4:"Without partner"}
MAR_PNS13 = {1:"With partner",2:"Without partner",3:"Without partner",
             4:"Without partner",5:"Without partner"}
MAR_PNS19 = {1:"With partner",2:"Without partner",
             3:"Without partner",4:"Without partner"}
INS_ELSI  = {0:"No",1:"Yes"}
INS_PNS   = {1:"Yes",2:"No"}
ZONE_R    = {1:"Urban",2:"Rural"}
REG_ELSI  = {1:"North",2:"Northeast",3:"Southeast",4:"South",5:"Central-West"}
UF_REG    = {11:"North",12:"North",13:"North",14:"North",15:"North",
             16:"North",17:"North",
             21:"Northeast",22:"Northeast",23:"Northeast",24:"Northeast",
             25:"Northeast",26:"Northeast",27:"Northeast",28:"Northeast",
             29:"Northeast",31:"Southeast",32:"Southeast",33:"Southeast",
             35:"Southeast",41:"South",42:"South",43:"South",
             50:"Central-West",51:"Central-West",52:"Central-West",
             53:"Central-West"}
EDUC_PNS  = {1:"Illiterate",2:"Incomplete Elem",3:"Complete Elem",
             4:"Incomplete Elem",5:"Complete Elem",
             6:"Higher Educ",7:"Higher Educ"}

def educ_elsi(v):
    if pd.isna(v): return np.nan
    v = int(v)
    if v==1: return "Illiterate"
    if 2<=v<=8: return "Incomplete Elem"
    if 9<=v<=13: return "Complete Elem"
    if 14<=v<=18: return "Higher Educ"
    return np.nan

CAT_FEAT = ["age_group","region","zone","marital","ethnicity",
            "education","insurance"]
NUM_FEAT = ["n_residents"]
ALL_FEAT = CAT_FEAT + NUM_FEAT

# ==============================================================================
# DATA PREP
# ==============================================================================
def prep_elsi(path, wave):
    df, _ = pyreadstat.read_dta(str(path), encoding="latin1")
    o = pd.DataFrame()
    o["age"]       = clean(df["idade"])
    o["sex"]       = pd.to_numeric(df["sexo"], errors="coerce")
    o["ethnicity"] = clean(df["e9"]).map(ETHNIC_ELSI_FULL)
    o["marital"]   = clean(df["e7"]).map(MAR_ELSI)
    o["zone"]      = clean(df["zona"]).map(ZONE_R)
    o["region"]    = clean(df["regiao"]).map(REG_ELSI)
    o["education"] = clean(df["e22"]).apply(educ_elsi)
    o["insurance"] = clean(df["u1"]).map(INS_ELSI)
    res = "ar6" if wave==1 else "nmoradores"
    o["n_residents"] = clean(df[res]).clip(upper=10)
    m13 = clean(df["m13"])
    o["target_pap"] = m13.isin({2,3}).astype("Int64")
    o["target_pap"] = o["target_pap"].where(m13.isin({1,2,3}), np.nan)
    m15 = clean(df["m15"])
    if wave in [1,2]:
        o["target_mamo"] = m15.isin({1,5}).astype("Int64")
    else:
        o["target_mamo"] = m15.isin({4,5}).astype("Int64")
    o["target_mamo"] = o["target_mamo"].where(m15.isin({1,2,3,4,5}), np.nan)
    return o[(o["sex"]==0) & (o["age"]>=50)].copy()

def prep_pns(path, sep, year):
    df = pd.read_csv(path, sep=sep, encoding="latin1", low_memory=False)
    o  = pd.DataFrame()
    o["age"]       = clean(df["C008"])
    o["sex"]       = clean(df["C006"])
    o["ethnicity"] = clean(df["C009"]).map(ETHNIC_PNS_FULL)
    o["zone"]      = clean(df["V0026"]).map(ZONE_R)
    o["region"]    = pd.to_numeric(df["V0001"],errors="coerce").map(UF_REG)
    o["education"] = clean(df["VDD004A"]).map(EDUC_PNS)
    o["n_residents"]= clean(df["V0022"]).clip(upper=10)
    if year==2013:
        o["marital"]  = clean(df["C011"]).map(MAR_PNS13)
        o["insurance"]= clean(df["I001"]).map(INS_PNS)
        pap = clean(df["R001"])
        o["target_pap"] = pap.isin([4,5]).astype("Int64")
        o["target_pap"] = o["target_pap"].where(pap.isin([1,2,3,4,5]), np.nan)
        r002 = pd.to_numeric(df["R002"], errors="coerce")
        o["target_pap"] = o["target_pap"].where(r002!=12, np.nan)
        md=clean(df["R015"]); mw=clean(df["R017"])
        o["target_mamo"] = ((md==2)|mw.isin([3,4])).astype("Int64")
        o["target_mamo"] = o["target_mamo"].where(md.isin([1,2])|mw.isin([1,2,3,4]),np.nan)
    else:
        o["marital"]  = clean(df["C011"]).map(MAR_PNS19)
        o["insurance"]= clean(df["I00102"]).map(INS_PNS)
        pap = clean(df["R00101"])
        o["target_pap"] = pap.isin([4,5]).astype("Int64")
        o["target_pap"] = o["target_pap"].where(pap.isin([1,2,3,4,5]), np.nan)
        r002010 = pd.to_numeric(df["R002010"], errors="coerce")
        o["target_pap"] = o["target_pap"].where(r002010!=12, np.nan)
        md=clean(df["R015"]); mw=clean(df["R01701"])
        o["target_mamo"] = ((md==2)|mw.isin([3,4])).astype("Int64")
        o["target_mamo"] = o["target_mamo"].where(md.isin([1,2])|mw.isin([1,2,3,4]),np.nan)
    return o[(o["sex"]==2) & (o["age"]>=50)].copy()

# ==============================================================================
# DESCRIPTIVE FUNCTION
# ==============================================================================
def get_Xy_desc(df, target, age_min, age_max):
    s = df[(df["age"]>=age_min) & (df["age"]<=age_max)].copy()
    s["age_group"] = pd.cut(s["age"], bins=[age_min-1,59,age_max],
                            labels=["50-59",f"60-{age_max}"]).astype(str)
    return s[ALL_FEAT+[target]].dropna()

def npct(series, val):
    n_valid = series.notna().sum()
    n_val   = (series==val).sum()
    if n_valid == 0: return "—"
    return f"{n_val:,} ({n_val/n_valid*100:.1f}%)"

def mssd(series):
    return f"{series.mean():.1f} ± {series.std():.1f}"

def desc_col(df, target, age_min, age_max, label):
    sub = get_Xy_desc(df, target, age_min, age_max)
    n   = len(sub)
    pos = int(sub[target].sum())
    return {
        "label":   label,
        "n":       n,
        "n_fmt":   f"{n:,}",
        "nonadh":  f"{pos:,} ({pos/n*100:.1f}%)" if n>0 else "—",
        "age5059": npct(sub["age_group"], "50-59"),
        "urban":   npct(sub["zone"],      "Urban"),
        "partner": npct(sub["marital"],   "With partner"),
        "insured": npct(sub["insurance"], "Yes"),
        "white":   npct(sub["ethnicity"], "White"),
        "black":   npct(sub["ethnicity"], "Black"),
        "brown":   npct(sub["ethnicity"], "Brown"),
        "other":   npct(sub["ethnicity"], "Other"),
        "illit":   npct(sub["education"], "Illiterate"),
        "incomp":  npct(sub["education"], "Incomplete Elem"),
        "compl":   npct(sub["education"], "Complete Elem"),
        "higher":  npct(sub["education"], "Higher Educ"),
        "north":   npct(sub["region"],    "North"),
        "ne":      npct(sub["region"],    "Northeast"),
        "se":      npct(sub["region"],    "Southeast"),
        "south":   npct(sub["region"],    "South"),
        "cw":      npct(sub["region"],    "Central-West"),
        "resid":   mssd(sub["n_residents"]),
    }

# ==============================================================================
# LOAD DATA
# ==============================================================================
print("Loading datasets...")
w1    = prep_elsi(ELSI_W1, 1)
w2    = prep_elsi(ELSI_W2, 2)
w3    = prep_elsi(ELSI_W3, 3)
pns13 = prep_pns(PNS13, ";", 2013)
pns19 = prep_pns(PNS19, ",", 2019)
print("Done")

# ==============================================================================
# BUILD DESCRIPTIVE COLUMNS
# ==============================================================================
DATASETS_PAP = [
    desc_col(w1,    "target_pap",  50, 64, "ELSI W1\n(2015–16)"),
    desc_col(w2,    "target_pap",  50, 64, "ELSI W2\n(2019–21)"),
    desc_col(w3,    "target_pap",  50, 64, "ELSI W3\n(2023–24)"),
    desc_col(pns13, "target_pap",  50, 64, "PNS 2013"),
    desc_col(pns19, "target_pap",  50, 64, "PNS 2019"),
]
DATASETS_MAMO = [
    desc_col(w1,    "target_mamo", 50, 69, "ELSI W1\n(2015–16)"),
    desc_col(w2,    "target_mamo", 50, 69, "ELSI W2\n(2019–21)"),
    desc_col(w3,    "target_mamo", 50, 69, "ELSI W3\n(2023–24)"),
    desc_col(pns13, "target_mamo", 50, 69, "PNS 2013"),
    desc_col(pns19, "target_mamo", 50, 69, "PNS 2019"),
]

# Print summary
print("\n=== PANEL A — PAP SMEAR ===")
for d in DATASETS_PAP:
    print(f"  {d['label'].replace(chr(10),' ')}: n={d['n']:,} | non-adh={d['nonadh']}")
print("\n=== PANEL B — MAMMOGRAPHY ===")
for d in DATASETS_MAMO:
    print(f"  {d['label'].replace(chr(10),' ')}: n={d['n']:,} | non-adh={d['nonadh']}")

# ==============================================================================
# WORD TABLE
# ==============================================================================
VARS = [
    ("Non-adherent, n (%)",          "nonadh",  True,  False),
    ("Age group",                    None,      False, True),
    ("  50–59 years",                "age5059", False, False),
    ("Area of residence",            None,      False, True),
    ("  Urban",                      "urban",   False, False),
    ("Marital status",               None,      False, True),
    ("  With partner",               "partner", False, False),
    ("Health insurance",             None,      False, True),
    ("  Yes",                        "insured", False, False),
    ("Race/ethnicity",               None,      False, True),
    ("  White",                      "white",   False, False),
    ("  Black",                      "black",   False, False),
    ("  Brown",                      "brown",   False, False),
    ("  Other\u1d43",               "other",   False, False),
    ("Education",                    None,      False, True),
    ("  Higher education",           "higher",  False, False),
    ("  Complete elementary",        "compl",   False, False),
    ("  Incomplete elementary",      "incomp",  False, False),
    ("  Illiterate",                 "illit",   False, False),
    ("Geographic region",            None,      False, True),
    ("  North",                      "north",   False, False),
    ("  Northeast",                  "ne",      False, False),
    ("  Southeast",                  "se",      False, False),
    ("  South",                      "south",   False, False),
    ("  Central-West",               "cw",      False, False),
    ("Household size, mean \u00b1 SD","resid",   False, False),
]

def set_bg(cell, color):
    tc=cell._tc; pr=tc.get_or_add_tcPr()
    shd=OxmlElement("w:shd")
    shd.set(qn("w:val"),"clear"); shd.set(qn("w:color"),"auto")
    shd.set(qn("w:fill"),color); pr.append(shd)

def set_txt(cell, text, bold=False, size=8.5,
            align=WD_ALIGN_PARAGRAPH.CENTER, italic=False):
    cell.text=""
    p=cell.paragraphs[0]; p.alignment=align
    r=p.add_run(str(text))
    r.bold=bold; r.italic=italic
    r.font.size=Pt(size); r.font.name="Times New Roman"

def build_panel(doc, datasets, panel_label, footnote_rows):
    # Panel heading
    ph = doc.add_paragraph()
    ph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    ph.add_run(panel_label).bold = True

    ncols = 1 + len(datasets)
    t = doc.add_table(rows=2+len(VARS), cols=ncols)
    t.style = "Table Grid"

    # Column widths
    col_w = [Cm(5.8)] + [Cm(2.9)]*len(datasets)
    for ci,w in enumerate(col_w):
        for cell in t.columns[ci].cells:
            cell.width = w

    # Row 1: Role headers
    role_labels = ["Development","Development","Temporal\nvalidation",
                   "External\nvalidation\n(retrospective)",
                   "External\nvalidation\n(prospective)"]
    role_spans  = [2, None, 1, 1, 1]  # W1+W2 span
    hdr0 = t.rows[0].cells
    set_txt(hdr0[0], "", size=8)
    for ci, (d, rl) in enumerate(zip(datasets, role_labels), start=1):
        set_txt(hdr0[ci], rl, bold=True, size=8)
        set_bg(hdr0[ci], "E8E8E8")

    # Row 2: Dataset headers
    hdr1 = t.rows[1].cells
    set_txt(hdr1[0], "Variable", bold=True, size=8.5,
            align=WD_ALIGN_PARAGRAPH.LEFT)
    set_bg(hdr1[0], "D9D9D9")
    for ci,d in enumerate(datasets, start=1):
        lbl = d["label"] + f"\nn = {d['n']:,}"
        set_txt(hdr1[ci], lbl, bold=True, size=8)
        set_bg(hdr1[ci], "D9D9D9")

    # Data rows
    for ri, (var_label, key, bold_row, is_hdr) in enumerate(VARS, start=2):
        row = t.rows[ri].cells
        set_txt(row[0], var_label, bold=(bold_row or is_hdr),
                align=WD_ALIGN_PARAGRAPH.LEFT, size=8.5)
        if is_hdr:
            set_bg(row[0], "F2F2F2")
        for ci, d in enumerate(datasets, start=1):
            val = d.get(key, "") if key else ""
            set_txt(row[ci], val, bold=bold_row, size=8.5)
            if is_hdr:
                set_bg(row[ci], "F2F2F2")

    doc.add_paragraph()

def build_table1(doc):
    # Title
    tp = doc.add_paragraph()
    tp.alignment = WD_ALIGN_PARAGRAPH.LEFT
    tr = tp.add_run(
        "Table 1. Sociodemographic characteristics of women included "
        "in the development, temporal validation, and external validation "
        "datasets, by screening outcome.")
    tr.bold=True; tr.font.size=Pt(10); tr.font.name="Times New Roman"
    doc.add_paragraph()

    build_panel(doc, DATASETS_PAP,
                "Panel A — Pap smear (women aged 50–64 years)", [])
    build_panel(doc, DATASETS_MAMO,
                "Panel B — Mammography (women aged 50–69 years)", [])

    # Footnote
    fn_txt = (
        "Values are n (%) unless otherwise stated. "
        "Non-adherence defined as: Pap smear performed >3 years ago or never "
        "(ELSI-Brazil: m13 codes 2–3; PNS: R001/R00101 codes 4–5); "
        "mammography performed >2 years ago or never "
        "(ELSI-Brazil W1/W2: m15 codes 1 and 5; W3: codes 4–5; "
        "PNS: R015=2 or R017/R01701 codes 3–4). "
        "Women reporting hysterectomy as the reason for Pap smear non-adherence "
        "(PNS 2013: R002=12; PNS 2019: R002010=12) were excluded from Pap smear samples. "
        "\u1d43 Other includes Yellow and Indigenous categories, collapsed due to small cell sizes. "
        "ELSI W1+W2 constitute the development (training) dataset; "
        "ELSI W3 the temporal validation dataset; "
        "PNS 2013 the retrospective and PNS 2019 the prospective external validation datasets. "
        "Ecological validation data: PAINEL-Oncologia/DATASUS 2016\u20132024, 25 Brazilian states "
        "(Amap\u00e1 excluded due to implausible staging completeness <12%; "
        "Federal District excluded due to referral centre profile); "
        "breast cancer (C50): 447,134 cases staged; "
        "cervical cancer (C53): 126,415 cases staged."
    )
    p = doc.add_paragraph()
    r = p.add_run(fn_txt)
    r.font.size=Pt(8); r.font.name="Times New Roman"
    p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY

# New doc
doc = Document()
for sec in doc.sections:
    sec.top_margin=Cm(2); sec.bottom_margin=Cm(2)
    sec.left_margin=Cm(2.5); sec.right_margin=Cm(2.5)

build_table1(doc)
doc.save(OUT_TAB/"table1_descriptive.docx")
print(f"\nSaved: table1_descriptive.docx")

# CSV backup
rows = []
for panel, datasets in [("Pap smear (50-64y)", DATASETS_PAP),
                         ("Mammography (50-69y)", DATASETS_MAMO)]:
    for d in datasets:
        for var_label, key, _, _ in VARS:
            if key:
                rows.append({"Panel":panel,"Dataset":d["label"].replace("\n"," "),
                             "Variable":var_label,"Value":d.get(key,"")})
pd.DataFrame(rows).to_csv(OUT_TAB/"table1_descriptive.csv", index=False)
print("Saved: table1_descriptive.csv")
