# =============================================================================
# MASTER ANALYSIS SCRIPT — R
# Cancer Screening Non-Adherence | ELSI W1+W2 Training Design
# =============================================================================
# Estrutura de pastas esperada (relativa a Codes/R/):
#   ../../Data/                     — dados brutos
#   ../../Outputs_v3/Python_tables/    — CSVs gerados pelo Python
#   ../../Outputs_v3/R_figures/        — figuras geradas aqui
#   ../../Outputs_v3/R_tables/         — tabelas Word geradas aqui
#
# Rodar: setwd("Final_5_samples/Codes/R"); source("master_analysis.R")
# =============================================================================

# ==============================================================================
# PACOTES
# ==============================================================================
pkgs <- c("rms","Hmisc","dplyr","ggplot2","ggrepel",
          "patchwork","stringr","sf","haven",
          "DiagrammeR","DiagrammeRsvg","rsvg")
for (p in pkgs)
  if (!requireNamespace(p, quietly=TRUE)) install.packages(p)

suppressPackageStartupMessages({
  library(rms);    library(Hmisc)
  library(dplyr);  library(ggplot2); library(ggrepel)
  library(patchwork); library(stringr); library(sf)
  library(haven)
  library(DiagrammeR); library(DiagrammeRsvg); library(rsvg)
})

options(contrasts = c("contr.treatment","contr.treatment"))

# ==============================================================================
# CAMINHOS
# ==============================================================================
OUT_NAME <- Sys.getenv("OUT_NAME", "")
if (OUT_NAME == "") stop("Defina OUT_NAME antes (ex.: Sys.setenv(OUT_NAME=\"Outputs_v3_edu4\")) para não sobrescrever outra análise.")
PY_TAB  <- file.path("../..", OUT_NAME, "Python_tables")
R_FIG   <- file.path("../..", OUT_NAME, "R_figures")
R_TAB   <- file.path("../..", OUT_NAME, "R_tables")
DATA    <- "../../Data"

dir.create(R_FIG, recursive=TRUE, showWarnings=FALSE)
dir.create(R_TAB, recursive=TRUE, showWarnings=FALSE)

cat("=== MASTER ANALYSIS ===\n")
cat("Python tables:", PY_TAB, "\n")
cat("R figures:    ", R_FIG,  "\n\n")

# ==============================================================================
# SEÇÃO 1 — NOMOGRAMAS (Pap smear + Mammography)
# Training data: ELSI W1+W2 pool
# ==============================================================================
cat("[ 1/4 ] Nomogramas...\n")

nomogram_outcome <- function(label, age_range, guideline, cid) {

  csv_file <- file.path(PY_TAB, paste0("nomogram_train_", label, "_ELSI.csv"))
  if (!file.exists(csv_file)) {
    cat("  SKIP:", csv_file, "not found — run export_nomogram_data in Python first\n")
    return(invisible(NULL))
  }

  df <- read.csv(csv_file)
  cat(sprintf("  %s: n=%s | prev=%.1f%%\n",
              label, format(nrow(df), big.mark=","),
              mean(df$target)*100))

  cat_vars <- c("Zone","Marital_status","Health_insurance",
                "Age_group","Region","Ethnicity","Education")
  df[cat_vars] <- lapply(df[cat_vars], as.factor)

  df$Zone             <- relevel(df$Zone,             ref="Urban")
  df$Marital_status   <- relevel(df$Marital_status,   ref="With partner")
  df$Health_insurance <- relevel(df$Health_insurance, ref="Yes")
  df$Region           <- relevel(df$Region,           ref="Southeast")
  df$Ethnicity        <- relevel(df$Ethnicity,        ref="White")
  df$Education        <- relevel(df$Education,        ref="Complete elementary to secondary")
  age_ref             <- paste0(strsplit(age_range,"-")[[1]][1], "-59")
  df$Age_group        <- relevel(df$Age_group,        ref=age_ref)

  df$N_residents <- pmin(as.numeric(df$N_residents), 10)

  dd <<- datadist(df); options(datadist="dd")

  f <- lrm(target ~ Zone + Marital_status + Health_insurance +
              Age_group + Region + Ethnicity + Education + N_residents,
            data=df, x=TRUE, y=TRUE)

  cat(sprintf("  C-statistic: %.3f\n", f$stats["C"]))
  cat("  OR:\n"); print(round(exp(coef(f)[-1]),3))

  # Nomogram
  nom_file <- file.path(R_FIG, paste0(label, "_nomogram_ELSI.png"))
  png(nom_file, width=2800, height=2000, res=220)
  plot(nomogram(f, fun=plogis, funlabel="P(non-adherent)",
                N_residents=1:10),
       cex.axis=0.85, lmgp=0.25)
  title(paste0(gsub("_"," ",label),
               " \u2014 Non-adherence nomogram\nELSI W1+W2 training data"),
        cex.main=1.1)
  dev.off()
  cat("  Saved:", nom_file, "\n")

  # Calibration
  cal_file <- file.path(R_FIG, paste0(label, "_calibration_ELSI.png"))
  png(cal_file, width=1400, height=1100, res=150)
  tryCatch({
    cal <- calibrate(f, method="boot", B=200)
    plot(cal, main=paste0(gsub("_"," ",label),
                          " \u2014 Calibration (bootstrap B=200)"))
  }, error=function(e) plot.new())
  dev.off()
  cat("  Saved:", cal_file, "\n")
}

nomogram_outcome("Pap_smear",   "50-64", "INCA 2016",    "C53")
nomogram_outcome("Mammography", "50-69", "MS/INCA biennial",  "C50")

# ==============================================================================
# SEÇÃO 2 — MAPAS COROPLÉTICOS
# Inputs: risk_scores_uf.csv + estadiamento_uf_separado.csv + BR_UF_2022.shp
# ==============================================================================
cat("\n[ 2/4 ] Mapas coropléticos...\n")

shp_path   <- file.path(DATA, "BR_UF_2022.shp")
risk_path  <- file.path(PY_TAB, "risk_scores_uf.csv")
stage_path <- file.path(DATA,   "estadiamento_uf_separado.csv")

if (!file.exists(shp_path)) {
  cat("  SKIP: BR_UF_2022.shp não encontrado em", DATA, "\n")
} else if (!file.exists(risk_path)) {
  cat("  SKIP: risk_scores_uf.csv não encontrado — rode export_risk_scores_v2.py\n")
} else if (!file.exists(stage_path)) {
  cat("  SKIP: estadiamento_uf_separado.csv não encontrado\n")
} else {

  shp <- st_read(shp_path, quiet=TRUE)
  shp <- st_transform(shp, crs=4674)

  df_risk  <- read.csv(risk_path)
  UF_SIGLA_MAP <- c(
    "11"="RO","12"="AC","13"="AM","14"="RR","15"="PA",
    "16"="AP","17"="TO","21"="MA","22"="PI","23"="CE",
    "24"="RN","25"="PB","26"="PE","27"="AL","28"="SE",
    "29"="BA","31"="MG","32"="ES","33"="RJ","35"="SP",
    "41"="PR","42"="SC","43"="RS","50"="MS","51"="MT",
    "52"="GO","53"="DF")

  # Long -> wide: one row per UF, columns per cancer type
  df_stage <- read.csv(stage_path) |>
    mutate(uf_str = str_pad(as.character(UF_cod), 2, "left", "0"),
           sigla  = UF_SIGLA_MAP[uf_str]) |>
    tidyr::pivot_wider(
      id_cols     = c(UF_cod, sigla),
      names_from  = cancer,
      values_from = prop_tardio,
      names_prefix= "prop_tardio_"
    )

  # Convert risk to percentage
  df_risk <- df_risk |>
    mutate(risk_pap_pct  = risk_pap  * 100,
           risk_mamo_pct = risk_mamo * 100)

  shp_data <- shp |>
    left_join(df_risk  |> select(sigla, risk_pap_pct, risk_mamo_pct),
              by=c("SIGLA_UF"="sigla")) |>
    left_join(df_stage |> select(sigla, prop_tardio_mama, prop_tardio_colo),
              by=c("SIGLA_UF"="sigla")) |>
    mutate(excluido = SIGLA_UF %in% c("AP","DF"))

  # Centroids for labels
  cents <- shp_data |>
    st_centroid() |>
    mutate(X=st_coordinates(geometry)[,1],
           Y=st_coordinates(geometry)[,2]) |>
    st_drop_geometry()

  tema <- theme_void(base_size=9) +
    theme(plot.title=element_text(size=8,face="bold",hjust=0),
          plot.subtitle=element_text(size=5.5,color="grey40",hjust=0),
          legend.position="bottom",
          legend.key.width=unit(2,"cm"),
          legend.key.height=unit(0.35,"cm"),
          legend.title=element_text(size=6.5,face="bold"),
          legend.text=element_text(size=5.5))

  make_map <- function(data, col, cmap, title, subtitle, leg_label) {
    vals <- data |> filter(!excluido) |> pull(!!sym(col))
    probs <- seq(0,1,length.out=5)
    brks  <- unique(round(quantile(vals,probs=probs,na.rm=TRUE),1))
    labs  <- paste0(brks,"%")
    vmin  <- min(brks); vmax <- max(brks)

    ggplot(data) +
      geom_sf(aes(fill=.data[[col]]),
              color="white",linewidth=0.3) +
      geom_sf(data=filter(data,excluido),
              fill="grey82",color="white",linewidth=0.3) +
      geom_text(data=cents,aes(x=X,y=Y,label=SIGLA_UF),
                size=2.0,color="grey15") +
      scale_fill_distiller(palette=cmap,direction=1,
                           na.value="grey82",name=leg_label,
                           breaks=brks,labels=labs,
                           limits=c(vmin,vmax),
                           oob=scales::squish) +
      labs(title=title,subtitle=subtitle) +
      coord_sf(datum=NA) + tema +
      guides(fill=guide_colorbar(title.position="top",
                                 barwidth=7,barheight=0.4))
  }

  # Ranges per row
  shp_incl <- shp_data |> filter(!excluido)
  risk_min  <- min(shp_incl$risk_pap_pct,  shp_incl$risk_mamo_pct, na.rm=TRUE)
  risk_max  <- max(shp_incl$risk_pap_pct,  shp_incl$risk_mamo_pct, na.rm=TRUE)
  stage_min <- min(shp_incl$prop_tardio_mama,shp_incl$prop_tardio_colo,na.rm=TRUE)
  stage_max <- max(shp_incl$prop_tardio_mama,shp_incl$prop_tardio_colo,na.rm=TRUE)

  pA <- make_map(shp_data,"risk_mamo_pct","YlOrRd",
                 "Predicted non-adherence risk \u2014 Mammography",
                 "XGBoost, ELSI W1+W2 \u2192 PNS 2019 (women 50\u201369y)",
                 "Mean predicted risk (%)")
  pB <- make_map(shp_data,"risk_pap_pct","YlOrRd",
                 "Predicted non-adherence risk \u2014 Pap smear",
                 "XGBoost, ELSI W1+W2 \u2192 PNS 2019 (women 50\u201364y)",
                 "Mean predicted risk (%)")
  pC <- make_map(shp_data,"prop_tardio_mama","OrRd",
                 "Late-stage diagnosis (III+IV) \u2014 Breast cancer (C50)",
                 "PAINEL-Oncologia / DATASUS 2016\u20132024, women 50\u201369y",
                 "Stage III+IV (%)")
  pD <- make_map(shp_data,"prop_tardio_colo","OrRd",
                 "Late-stage diagnosis (III+IV) \u2014 Cervical cancer (C53)",
                 "PAINEL-Oncologia / DATASUS 2016\u20132024, women 50\u201364y",
                 "Stage III+IV (%)")

  p_combined <- (pA + pC) / (pB + pD) +
    plot_annotation(
      caption=paste0(
        "Grey states (AP, DF) excluded from ecological correlation analysis.\n",
        "AP: implausible staging completeness. DF: atypical referral centre.\n",
        "Colour scales consistent within each row. Quintile breaks."),
      theme=theme(plot.caption=element_text(size=4.5,colour="grey50",hjust=0)))

  map_out <- file.path(R_FIG,"maps_screening_combined.png")
  ggsave(map_out, p_combined, width=16, height=18, dpi=600, units="cm")
  ggsave(file.path(R_FIG,"maps_screening_combined.pdf"),
         p_combined, width=16, height=18, units="cm")
  cat("  Saved: maps_screening_combined.png + .pdf\n")

  # Export data for maps (Excel-compatible)
  shp_data |> st_drop_geometry() |>
    select(SIGLA_UF, risk_pap_pct, risk_mamo_pct,
           prop_tardio_mama, prop_tardio_colo) |>
    write.csv(file.path(R_TAB,"map_data_by_state.csv"), row.names=FALSE)
  cat("  Saved: map_data_by_state.csv\n")
}

# ==============================================================================
# SEÇÃO 3 — SCATTER ECOLÓGICO + CORRELAÇÃO
# ==============================================================================
cat("\n[ 3/4 ] Scatter ecológico...\n")

if (exists("shp_data") && !is.null(shp_data)) {

  UF_SIGLA <- c(
    "11"="RO","12"="AC","13"="AM","14"="RR","15"="PA","16"="AP","17"="TO",
    "21"="MA","22"="PI","23"="CE","24"="RN","25"="PB","26"="PE","27"="AL",
    "28"="SE","29"="BA","31"="MG","32"="ES","33"="RJ","35"="SP",
    "41"="PR","42"="SC","43"="RS","50"="MS","51"="MT","52"="GO","53"="DF")

  df_eco <- shp_data |>
    st_drop_geometry() |>
    filter(!excluido) |>
    select(sigla=SIGLA_UF, risk_pap_pct, risk_mamo_pct,
           prop_tardio_mama, prop_tardio_colo) |>
    filter(!is.na(risk_mamo_pct) & !is.na(prop_tardio_mama))

  # Correlações
  cor_mama <- cor.test(df_eco$risk_mamo_pct, df_eco$prop_tardio_mama,
                       method="spearman", exact=FALSE)
  cor_colo <- cor.test(df_eco$risk_pap_pct,  df_eco$prop_tardio_colo,
                       method="spearman", exact=FALSE)

  cat(sprintf("  Mammography: rho=%.3f p=%.3f n=%d\n",
              cor_mama$estimate, cor_mama$p.value, nrow(df_eco)))
  cat(sprintf("  Pap smear:   rho=%.3f p=%.3f n=%d\n",
              cor_colo$estimate, cor_colo$p.value, nrow(df_eco)))

  tema_scatter <- theme_classic(base_size=10) +
    theme(plot.title=element_text(size=10,face="bold"),
          axis.title=element_text(size=9),
          axis.text=element_text(size=8))

  make_scatter <- function(data, x_col, y_col, title,
                           xlab, ylab, rho, pval) {
    pval_str <- ifelse(pval<0.001,"p < 0.001",
                       sprintf("p = %.3f",pval))
    ann <- sprintf("Spearman \u03c1 = %.3f\n%s", rho, pval_str)

    ggplot(data, aes(x=.data[[x_col]], y=.data[[y_col]])) +
      geom_smooth(method="lm",se=TRUE,color="#1565C0",
                  fill="#BBDEFB",alpha=0.3,linewidth=0.8) +
      geom_point(size=2.5,color="#1565C0",alpha=0.8) +
      geom_text_repel(aes(label=sigla),size=2.8,
                      color="grey30",max.overlaps=20) +
      annotate("text",x=Inf,y=-Inf,hjust=1.1,vjust=-0.5,
               label=ann,size=3,fontface="italic",color="#1565C0") +
      labs(title=title,x=xlab,y=ylab) +
      tema_scatter
  }

  sA <- make_scatter(df_eco,"risk_mamo_pct","prop_tardio_mama",
                     "A  Mammography",
                     "Mean predicted non-adherence risk (%)",
                     "Stage III+IV at diagnosis (%)",
                     cor_mama$estimate, cor_mama$p.value)

  sB <- make_scatter(df_eco,"risk_pap_pct","prop_tardio_colo",
                     "B  Pap smear",
                     "Mean predicted non-adherence risk (%)",
                     "Stage III+IV at diagnosis (%)",
                     cor_colo$estimate, cor_colo$p.value)

  p_scatter <- sA / sB

  sc_out <- file.path(R_FIG,"scatter_ecological.png")
  ggsave(sc_out, p_scatter, width=8, height=16, dpi=600, units="cm")
  ggsave(file.path(R_FIG,"scatter_ecological.pdf"),
         p_scatter, width=8, height=16, units="cm")
  cat("  Saved: scatter_ecological.png + .pdf\n")

} else {
  cat("  SKIP: shp_data not available (maps section was skipped)\n")
}

# ==============================================================================
# SEÇÃO 4 — FLOWCHART ATTRITION (novo design ELSI W1+W2+W3 + PNS 2013 + 2019)
# ==============================================================================
cat("\n[ 4/4 ] Flowchart attrition...\n")

# (As funções de attrition em R foram removidas: os números vêm de attrition_counts.csv,
#  gerado por attrition_counts_v3.py com as mesmas regras do common.py.)

fmt_n <- function(x) format(round(x), big.mark=",", scientific=FALSE)

# Load attrition counts from Python (same filters as pipeline)
  att_path <- file.path(PY_TAB, "attrition_counts.csv")

  if (!file.exists(att_path)) {
    cat("  SKIP: attrition_counts.csv not found\n")
    cat("  Run: python attrition_counts.py first\n")
  } else {
  att <- read.csv(att_path)
  get_n <- function(ds, step) {
    v <- att$n[att$dataset==ds & att$step==step]
    if (length(v)==0) return(0L); as.integer(v[1])
  }
  make_steps <- function(ds) list(
    raw=get_n(ds,"raw"), women=get_n(ds,"women"),
    age50=get_n(ds,"age50"), zone=get_n(ds,"zone"),
    pap_avail=get_n(ds,"pap_avail"),
    mamo_avail=get_n(ds,"mamo_avail"),
    complete=get_n(ds,"complete"),
    n_pap=get_n(ds,"n_pap"), n_mamo=get_n(ds,"n_mamo")
  )
  s_w1  <- make_steps("ELSI_W1")
  s_w2  <- make_steps("ELSI_W2")
  s_w3  <- make_steps("ELSI_W3")
  s_p13 <- make_steps("PNS_2013")
  s_p19 <- make_steps("PNS_2019")
  cat(sprintf("  ELSI W1:  Pap=%s | Mamo=%s\n",
              fmt_n(s_w1$n_pap), fmt_n(s_w1$n_mamo)))
  cat(sprintf("  ELSI W2:  Pap=%s | Mamo=%s\n",
              fmt_n(s_w2$n_pap), fmt_n(s_w2$n_mamo)))
  cat(sprintf("  ELSI W3:  Pap=%s | Mamo=%s\n",
              fmt_n(s_w3$n_pap), fmt_n(s_w3$n_mamo)))
  cat(sprintf("  PNS 2013: Pap=%s | Mamo=%s\n",
              fmt_n(s_p13$n_pap), fmt_n(s_p13$n_mamo)))
  cat(sprintf("  PNS 2019: Pap=%s | Mamo=%s\n",
              fmt_n(s_p19$n_pap), fmt_n(s_p19$n_mamo)))

  # Montar DOT
  dot <- paste0('
digraph flowchart {
  graph [layout=dot, rankdir=TB, splines=ortho,
         nodesep=0.35, ranksep=0.6, fontname="Helvetica"]
  node [shape=rectangle, style=filled, fillcolor=white,
        color=black, penwidth=1, fontname="Helvetica",
        fontsize=9.5, margin="0.12,0.07", width=1.6]
  edge [color=black, arrowsize=0.65, penwidth=0.75]

  node [fillcolor="#333333", fontcolor=white, fontsize=10.5,
        style=filled, penwidth=2, width=1.6]
  h1 [label="ELSI-Brazil W1\n(2015\u201316)"]
  h2 [label="ELSI-Brazil W2\n(2019\u201321)"]
  h3 [label="ELSI-Brazil W3\n(2023\u201324)"]
  h4 [label="PNS 2013"]
  h5 [label="PNS 2019"]

  node [fillcolor=white, fontcolor=black, fontsize=9,
        penwidth=0.85, style=filled, width=1.6]

  # ELSI W1
  a1 [label="Raw sample\nn = ', fmt_n(s_w1$raw), '"]
  a2 [label="Women \u2265 50 years\nn = ', fmt_n(s_w1$women), '"]
  a3 [label="Outcomes available\nPap: n = ', fmt_n(s_w1$pap_avail), ' \nMamo: n = ', fmt_n(s_w1$mamo_avail), '"]
  a6 [label="Analytical sample\nPap (50\u201364y): n = ', fmt_n(s_w1$n_pap), '\nMamo (50\u201369y): n = ', fmt_n(s_w1$n_mamo), '"
      fillcolor="#DDDDDD" penwidth=2]

  # ELSI W2
  b1 [label="Raw sample\nn = ', fmt_n(s_w2$raw), '"]
  b2 [label="Women \u2265 50 years\nn = ', fmt_n(s_w2$women), '"]
  b3 [label="Outcomes available\nPap: n = ', fmt_n(s_w2$pap_avail), ' \nMamo: n = ', fmt_n(s_w2$mamo_avail), '"]
  b6 [label="Analytical sample\nPap (50\u201364y): n = ', fmt_n(s_w2$n_pap), '\nMamo (50\u201369y): n = ', fmt_n(s_w2$n_mamo), '"
      fillcolor="#DDDDDD" penwidth=2]

  # ELSI W3
  c1 [label="Raw sample\nn = ', fmt_n(s_w3$raw), '"]
  c2 [label="Women \u2265 50 years\nn = ', fmt_n(s_w3$women), '"]
  c3 [label="Outcomes available\nPap: n = ', fmt_n(s_w3$pap_avail), ' \nMamo: n = ', fmt_n(s_w3$mamo_avail), '"]
  c6 [label="Analytical sample\nPap (50\u201364y): n = ', fmt_n(s_w3$n_pap), '\nMamo (50\u201369y): n = ', fmt_n(s_w3$n_mamo), '"
      fillcolor="#DDDDDD" penwidth=2]

  # PNS 2013
  d1 [label="Raw sample\nn = ', fmt_n(s_p13$raw), '"]
  d2 [label="Women\nn = ', fmt_n(s_p13$women), '"]
  d3 [label="Women \u2265 50 years\nn = ', fmt_n(s_p13$zone), '"]
  d4 [label="Pap outcome available\nn = ', fmt_n(s_p13$pap_avail), '"]
  d5 [label="Mamo outcome available\nn = ', fmt_n(s_p13$mamo_avail), '"]
  d6 [label="Complete predictors\nn = ', fmt_n(s_p13$complete), '"]
  d7 [label="Analytical sample\nPap (50\u201364y): n = ', fmt_n(s_p13$n_pap), '\nMamo (50\u201369y): n = ', fmt_n(s_p13$n_mamo), '"
      fillcolor="#DDDDDD" penwidth=2]

  # PNS 2019
  e1 [label="Raw sample\nn = ', fmt_n(s_p19$raw), '"]
  e2 [label="Women\nn = ', fmt_n(s_p19$women), '"]
  e3 [label="Women \u2265 50 years\nn = ', fmt_n(s_p19$zone), '"]
  e4 [label="Pap outcome available\nn = ', fmt_n(s_p19$pap_avail), '"]
  e5 [label="Mamo outcome available\nn = ', fmt_n(s_p19$mamo_avail), '"]
  e6 [label="Complete predictors\nn = ', fmt_n(s_p19$complete), '"]
  e7 [label="Analytical sample\nPap (50\u201364y): n = ', fmt_n(s_p19$n_pap), '\nMamo (50\u201369y): n = ', fmt_n(s_p19$n_mamo), '"
      fillcolor="#DDDDDD" penwidth=2]

  # Exclusion nodes
  node [shape=rectangle, style="filled,dashed", fillcolor=white,
        fontcolor="#333333", fontsize=8, penwidth=0.7, width=1.3]

  ea2 [label="Excl: ', fmt_n(s_w1$raw-s_w1$women), ' men"]
  ea3 [label="Excl: ', fmt_n(s_w1$age50-s_w1$pap_avail), ' (Pap)\n', fmt_n(s_w1$age50-s_w1$mamo_avail), ' (Mamo)\nmissing outcomes"]
  ea5 [label="Excl: ', fmt_n(s_w1$pap_avail-s_w1$n_pap), ' (Pap)\n', fmt_n(s_w1$mamo_avail-s_w1$n_mamo), ' (Mamo)\nmissing pred. + age limit"]

  eb2 [label="Excl: ', fmt_n(s_w2$raw-s_w2$women), ' men"]
  eb3 [label="Excl: ', fmt_n(s_w2$age50-s_w2$pap_avail), ' (Pap)\n', fmt_n(s_w2$age50-s_w2$mamo_avail), ' (Mamo)\nmissing outcomes"]
  eb5 [label="Excl: ', fmt_n(s_w2$pap_avail-s_w2$n_pap), ' (Pap)\n', fmt_n(s_w2$mamo_avail-s_w2$n_mamo), ' (Mamo)\nmissing pred. + age limit"]

  ec2 [label="Excl: ', fmt_n(s_w3$raw-s_w3$women), ' men"]
  ec3 [label="Excl: ', fmt_n(s_w3$age50-s_w3$pap_avail), ' (Pap)\n', fmt_n(s_w3$age50-s_w3$mamo_avail), ' (Mamo)\nmissing outcomes"]
  ec5 [label="Excl: ', fmt_n(s_w3$pap_avail-s_w3$n_pap), ' (Pap)\n', fmt_n(s_w3$mamo_avail-s_w3$n_mamo), ' (Mamo)\nmissing pred. + age limit"]

  ed2 [label="Excl: ', fmt_n(s_p13$raw-s_p13$women), ' men"]
  ed3 [label="Excl: ', fmt_n(s_p13$women-s_p13$zone), '\nage <50"]
  ed4 [label="Excl: ', fmt_n(s_p13$zone-s_p13$pap_avail), '\nmissing/hyst."]
  ed5 [label="Excl: ', fmt_n(s_p13$pap_avail-s_p13$mamo_avail), '\nmissing/unknown"]
  ed6 [label="Excl: ', fmt_n(s_p13$mamo_avail-s_p13$complete), '\nmissing pred."]
  ed7 [label="Outcome-specific samples:\nPap age \u226464 y; Mamo age \u226469 y"]

  ee2 [label="Excl: ', fmt_n(s_p19$raw-s_p19$women), ' men"]
  ee3 [label="Excl: ', fmt_n(s_p19$women-s_p19$zone), '\nage <50"]
  ee4 [label="Excl: ', fmt_n(s_p19$zone-s_p19$pap_avail), '\nmissing/hyst."]
  ee5 [label="Excl: ', fmt_n(s_p19$pap_avail-s_p19$mamo_avail), '\nmissing/unknown"]
  ee6 [label="Excl: ', fmt_n(s_p19$mamo_avail-s_p19$complete), '\nmissing pred."]
  ee7 [label="Outcome-specific samples:\nPap age \u226464 y; Mamo age \u226469 y"]

  { rank=same; h1;h2;h3;h4;h5 }
  { rank=same; a1;b1;c1;d1;e1; ea2;eb2;ec2;ed2;ee2 }
  { rank=same; a2;b2;c2;d2;e2; ea3;eb3;ec3;ed3;ee3 }
  { rank=same; a3;b3;c3;d3;e3; ea5;eb5;ec5;ed4;ee4 }
  { rank=same; d4;e4; ed5;ee5 }
  { rank=same; d5;e5; ed6;ee6 }
  { rank=same; a6;b6;c6;d6;e6;       ed7;ee7 }
  { rank=same;          d7;e7 }

  edge [style=solid, arrowhead=normal]
  h1->a1 h2->b1 h3->c1 h4->d1 h5->e1
  a1->a2 b1->b2 c1->c2 d1->d2 e1->e2
  a2->a3 b2->b3 c2->c3 d2->d3 e2->e3
  a3->a6 b3->b6 c3->c6 d3->d4 e3->e4
  d4->d5 e4->e5
  d5->d6 e5->e6
  d6->d7 e6->e7

  edge [style=dashed, arrowhead=open, color="#555555"]
  a1->ea2 a2->ea3 a3->ea5
  b1->eb2 b2->eb3 b3->eb5
  c1->ec2 c2->ec3 c3->ec5
  d1->ed2 d2->ed3 d3->ed4 d4->ed5 d5->ed6 d6->ed7
  e1->ee2 e2->ee3 e3->ee4 e4->ee5 e5->ee6 e6->ee7
}
')

  g       <- grViz(dot)
  svg_txt <- export_svg(g)
  fc_out  <- file.path(R_FIG,"flowchart_attrition_final.png")
  rsvg_png(charToRaw(svg_txt), fc_out, width=6000, height=3600)
  rsvg_pdf(charToRaw(svg_txt),
           file.path(R_FIG,"flowchart_attrition_final.pdf"))
  cat("  Saved: flowchart_attrition_final.png + .pdf\n")
}

# ==============================================================================
# RESUMO FINAL
# ==============================================================================
cat("\n=== CONCLUÍDO ===\n")
cat("Figures saved to:", R_FIG, "\n")
cat("Tables saved to: ", R_TAB, "\n")
