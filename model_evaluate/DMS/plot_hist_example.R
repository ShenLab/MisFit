library(ggplot2)
library(data.table)
library(ggpp)
source("../../../settings/plot_settings.R")

sigmoid = function(x) {
  return (1 / (1+exp(-x)))
}

data_all = fread("DMS_combined_label.txt.gz")
colnames(data_all)[which(colnames(data_all)=="model_damage")]="MisFit_D"
colnames(data_all)[which(colnames(data_all)=="model_selection")]="MisFit_S"
colnames(data_all)[which(colnames(data_all)=="CADD_phred")]="CADD"
data_all[, MisFit_S:=log10(sigmoid(MisFit_S))]
data_all[,target:=factor(target)]

example_methods = c("MisFit_D", "MisFit_S", 'AlphaMissense', 'CADD', 'REVEL')
example_genes = c("BRCA1", "PTEN", "TP53")

data_selected = data_all[is.element(Symbol, example_genes), ]
data_selected = data_selected[, c("target", "Symbol", example_methods), with = F]
data_selected = melt(data_selected, measure.vars = example_methods, variable.name = "method", value.name = "score")

dt_ROC = fread("AUC_combined_summary.txt")
colnames(dt_ROC)[which(colnames(dt_ROC)=="model_damage")]="MisFit_D"
colnames(dt_ROC)[which(colnames(dt_ROC)=="model_selection")]="MisFit_S"
colnames(dt_ROC)[which(colnames(dt_ROC)=="CADD_phred")]="CADD"
dt_ROC = dt_ROC[is.element(Symbol, example_genes), ]
dt_ROC = dt_ROC[, c("Symbol", example_methods), with = F]
dt_ROC = melt(dt_ROC, measure.vars = example_methods, variable.name = "method", value.name = "AUROC")


ggplot(data_selected) + geom_histogram(aes(x = score, fill = target), position="identity", alpha = 0.4) + 
  geom_text_npc(data = dt_ROC, aes(label = paste0("AUROC: ", round(AUROC, 3))), npcx = "left", npcy = "top", vjust = 1) +
  facet_grid(vars(Symbol), vars(method), scales = "free") +
  theme_nature() +
  scale_x_continuous(limits = c(NA, NA)) +
  scale_fill_manual(name = "", values = c("darkblue", "red"), labels = c("benign", "damaging"))
  

save_nature("figure/example_distr.pdf", hw_ratio = 1/2, width = "double")






