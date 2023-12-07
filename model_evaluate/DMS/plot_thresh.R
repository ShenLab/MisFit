library(ggplot2)
library(data.table)

thresh_df = fread("thresh_combined_summary.txt")
colnames(thresh_df)[which(colnames(thresh_df)=="model_damage")]="MisFit_D"
colnames(thresh_df)[which(colnames(thresh_df)=="model_selection")]="MisFit_S"
colnames(thresh_df)[which(colnames(thresh_df)=="CADD_phred")]="CADD"

data_all = fread("DMS_combined_label.txt.gz")
colnames(data_all)[which(colnames(data_all)=="model_damage")]="MisFit_D"
colnames(data_all)[which(colnames(data_all)=="model_selection")]="MisFit_S"
colnames(data_all)[which(colnames(data_all)=="CADD_phred")]="CADD"
data_all[,target:=factor(target)]

example_methods = c("MisFit_D", "MisFit_S", 'ESM-2', 'ESM-1b', 'AlphaMissense', 'EVE', 'PrimateAI', 'gMVP', 'CADD', 'REVEL', 'MPC')

thresh_df = fread("thresh_rank_combined_summary.txt")
colnames(thresh_df)[which(colnames(thresh_df)=="model_damage")]="MisFit_D"
colnames(thresh_df)[which(colnames(thresh_df)=="model_selection")]="MisFit_S"
colnames(thresh_df)[which(colnames(thresh_df)=="CADD_phred")]="CADD"

plist = list()

for (i in 1:length(example_methods)) {
  method = example_methods[i]
  data_all[, rank:=frank(get(method), na.last = "keep")]
  data_all[, rank:=rank / sum(!is.na(rank))]
  data_all[, (method):=rank]
  plist[[i]] = ggplot() + 
    geom_line(data = data_all, aes(x = .data[[method]], group = interaction(Symbol, target), color = target), stat="density", alpha=0.15) +
    geom_line(data = data_all, aes(x = .data[[method]], color = target), stat="density") +
    scale_color_manual(name = "", values = c("blue", "firebrick"), labels = c("benign", "damaging")) +
    scale_x_continuous(name = "", breaks = seq(0, 1, 0.2)) +
    geom_vline(data = thresh_df[Symbol!=""], aes(xintercept = .data[[method]]), color = "black", alpha = 0.15) +
    geom_vline(data = thresh_df[Symbol==""], aes(xintercept = .data[[method]]), color = "black") +
    theme_nature() +
    theme(panel.grid = element_blank()) +
    ggtitle(method)
}
arrange_nature(plotlist = plist, nrow = 4, ncol = 3, common.legend = T, labels = "", legend = "bottom")
save_nature("figure/thresh_rank.pdf", hw_ratio = 1/1.5, width = "double")






