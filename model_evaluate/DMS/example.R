library(ggplot2)
library(data.table)

source("../../../settings/plot_settings.R")

genes = c("BRCA1", "NUDT15", "ADRB2", "HRAS")
plist = list()
for (i in 1:length(genes)) {
  gene = genes[i]
  data = fread(paste0(gene, "_merged.txt.gz"))
  plist[[i]] = ggplot(data, aes(x = functional_score, fill = factor(target))) + geom_histogram() +
    theme_bw() +
    labs(x = "functional score", title = gene) +
    scale_fill_manual(name = "", values = c("blue", "firebrick", "grey"), labels = c("benign", "damaging", "uncertain"))
  
}

arrange_nature(plotlist = plist, common.legend = T)
save_nature("figure/example.pdf", hw_ratio = 1, width_ratio = 1)
