library(data.table)
library(stringr)
library(ggrepel)
source("../../../settings/plot_settings.R")

scores = c("MisFit_D", "MisFit_S", 'ESM-2', 'ESM-1b', 'AlphaMissense', 'EVE', 'PrimateAI', 'gMVP', 'CADD', 'REVEL', 'MPC')

ROC_all_2 = fread("Clinvar_summary_2_all.txt")
ROC_all_2 = ROC_all_2[by =="all"]
ROC_all_2[method=="CADD_phred", method:="CADD"]
ROC_all_2[method=="model_damage", method:="MisFit_D"]
ROC_all_2[method=="model_selection", method:="MisFit_S"]

ROC_all_2[, method:=factor(method, levels = scores)]

p1 = ggplot(ROC_all_2, aes(x = reorder(method, AUC), y = AUC, fill = is.element(method, c("MisFit_D", "MisFit_S")))) + 
  theme_nature() + 
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("grey70", "firebrick")) +
  scale_y_continuous(name = "ClinVar AUROC", breaks = seq(0, 1, 0.2), minor_breaks = c()) +
  coord_flip() +
  labs(x = "", title = "EVE genes") +
  theme(legend.position = "none")

# all genes
ROC_all_2 = fread("Clinvar_summary_2_exclude.txt")
ROC_all_2 = ROC_all_2[by =="all"]
ROC_all_2[method=="CADD_phred", method:="CADD"]
ROC_all_2[method=="model_damage", method:="MisFit_D"]
ROC_all_2[method=="model_selection", method:="MisFit_S"]

ROC_all_2[, method:=factor(method, levels = scores)]

p2 = ggplot(ROC_all_2, aes(x = reorder(method, AUC), y = AUC, fill = is.element(method, c("MisFit_D", "MisFit_S")))) + 
  theme_nature() + 
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("grey70", "firebrick")) +
  scale_y_continuous(name = "ClinVar AUROC", breaks = seq(0, 1, 0.2), minor_breaks = c()) +
  coord_flip() +
  labs(x = "", title = "all") +
  theme(legend.position = "none")


arrange_nature(p2, p1, common.legend = F)
save_nature("summary_clinvar.pdf", hw_ratio = 1/2)
