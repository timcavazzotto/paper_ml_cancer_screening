# Nomogram — Mammography | Training: ELSI W1+W2 | MS/INCA biennial (50-69y)
if (!requireNamespace("rms",quietly=TRUE)) install.packages("rms")
if (!requireNamespace("Hmisc",quietly=TRUE)) install.packages("Hmisc")
library(rms); library(Hmisc)
options(contrasts=c("contr.treatment","contr.treatment"))

df <- read.csv("../../Outputs_v3_edu4/Python_tables/nomogram_train_Mammography_ELSI.csv")
cat_vars <- c("Zone","Marital_status","Health_insurance",
              "Age_group","Region","Ethnicity","Education")
df[cat_vars] <- lapply(df[cat_vars], as.factor)
df$Zone             <- relevel(df$Zone,             ref="Urban")
df$Marital_status   <- relevel(df$Marital_status,   ref="With partner")
df$Health_insurance <- relevel(df$Health_insurance, ref="Yes")
df$Region           <- relevel(df$Region,           ref="Southeast")
df$Ethnicity        <- relevel(df$Ethnicity,        ref="White")
df$Education        <- relevel(df$Education,        ref="Complete elementary to secondary")
df$Age_group        <- relevel(df$Age_group,        ref="50-59")
dd <- datadist(df); options(datadist="dd")
f <- lrm(target ~ Zone + Marital_status + Health_insurance +
           Age_group + Region + Ethnicity + Education + N_residents,
         data=df, x=TRUE, y=TRUE)
print(f)
cat("
C-statistic:", f$stats["C"], "
")
print(round(exp(coef(f)[-1]),3))
png("../../Outputs_v3_edu4/R_figures/Mammography_nomogram_ELSI.png",width=2800,height=2000,res=220)
plot(nomogram(f,fun=plogis,funlabel="P(non-adherent)",N_residents=1:10),
     cex.axis=0.85,lmgp=0.25)
title("Mammography - Non-adherence nomogram
ELSI W1+W2 training data",cex.main=1.1)
dev.off()
pdf("../../Outputs_v3_edu4/R_figures/Mammography_nomogram_ELSI.pdf",width=11,height=8)
plot(nomogram(f,fun=plogis,funlabel="P(non-adherent)",N_residents=1:10),
     cex.axis=0.85,lmgp=0.25)
title("Mammography - Non-adherence nomogram
ELSI W1+W2 training data",cex.main=1.1)
dev.off()
png("../../Outputs_v3_edu4/R_figures/Mammography_calibration_ELSI.png",width=1400,height=1100,res=150)
plot(calibrate(f,method="boot",B=200),main="Mammography - Calibration")
dev.off()
cat("Done
")
