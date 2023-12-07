library(ggplot2)
library(data.table)
library(stringr)
library(ggrepel)
library(pROC)
library(PRROC)
library(ggpubr)
library(yardstick)
library(ggpp)
source("../../../settings/plot_settings.R")
scores = c("MisFit_D", "MisFit_S", 'ESM-2', 'ESM-1b', 'AlphaMissense', 'gMVP', 'EVE', 'PrimateAI', 'CADD', 'REVEL', 'MPC')
scores = rev(scores)
method_dt = data.table(method = scores)
method_dt[, method_type:="unsupervised"]
method_dt[is.element(method, c("CADD", "gMVP", "REVEL", "MPC")), method_type:="supervised"]

# spearman
corr_dt = fread("correlation_summary.txt")
corr_dt = corr_dt[num_AA_change>400]
#corr_dt = unique(corr_dt, by = "Symbol")
corr_dt[, geneset:="other"]
corr_dt[complete.cases(corr_dt), geneset:="EVE gene"]
colnames(corr_dt)[which(colnames(corr_dt)=="model_damage")]="MisFit_D"
colnames(corr_dt)[which(colnames(corr_dt)=="model_selection")]="MisFit_S"
colnames(corr_dt)[which(colnames(corr_dt)=="CADD_phred")]="CADD"

corr_dt2 = melt(corr_dt,
                measure.vars = colnames(corr_dt)[is.element(colnames(corr_dt), scores)], 
                id.vars = c('DMS', 'Symbol', 'num_AA_change', 'geneset'),
                variable.name = "method",
                variable.factor = T,
                value.name = "metric"
)
corr_dt2[, method:=factor(method, levels = scores)]

mean_corr = corr_dt2[geneset == "EVE gene", mean(metric, na.rm = T), by = .(method)]
colnames(mean_corr)[2] = "metric"
p1 = ggplot(corr_dt2[geneset == "EVE gene"], aes(y = method, x = metric)) +
  theme_nature() +
  geom_violin(fill = "lightblue", color = NA) +
  scale_x_continuous(expand = c(0, 0.1)) + 
  theme(panel.grid = element_blank()) +
  geom_jitter(color = "grey") +
  stat_summary(fun=mean, colour="firebrick", geom="point") +
  geom_text(data = mean_corr, aes(label = round(metric, 2), x = metric + 0.1), color = "firebrick") +
  labs(x = expression(spearman~rho), y = element_blank())
  
 
mean_corr = corr_dt2[method!="EVE", mean(metric, na.rm = T), by = .(method)]
colnames(mean_corr)[2] = "metric"
p2 = ggplot(corr_dt2[method!="EVE"], aes(y = method, x = metric)) +
  theme_nature() +
  geom_violin(fill = "lightblue", color = NA) +
  scale_x_continuous(expand = c(0, 0.1)) + 
  theme(panel.grid = element_blank()) +
  geom_jitter(color = "grey") +
  stat_summary(fun=mean, colour="firebrick", geom="point") +
  geom_text(data = mean_corr, aes(label = round(metric, 2), x = metric + 0.1), color = "firebrick") +
  labs(x = expression(spearman~rho), y = element_blank())

# AUC
corr_dt = fread("AUC_combined_summary.txt")
corr_dt[, geneset:="other"]
corr_dt[complete.cases(corr_dt), geneset:="EVE gene"]
colnames(corr_dt)[which(colnames(corr_dt)=="model_damage")]="MisFit_D"
colnames(corr_dt)[which(colnames(corr_dt)=="model_selection")]="MisFit_S"
colnames(corr_dt)[which(colnames(corr_dt)=="CADD_phred")]="CADD"

corr_dt2 = melt(corr_dt,
                measure.vars = colnames(corr_dt)[is.element(colnames(corr_dt), scores)], 
                id.vars = c('Symbol', 'num_AA_change', 'geneset'),
                variable.name = "method",
                variable.factor = T,
                value.name = "metric"
)
corr_dt2[, method:=factor(method, levels = scores)]

mean_corr = corr_dt2[geneset == "EVE gene", mean(metric, na.rm = T), by = .(method)]
colnames(mean_corr)[2] = "metric"
p3 = ggplot(corr_dt2[geneset == "EVE gene"], aes(y = method, x = metric)) +
  theme_nature() +
  geom_violin(fill = "lightblue", color = NA) +
  scale_x_continuous(expand = c(0, 0.1)) + 
  theme(panel.grid = element_blank()) +
  geom_jitter(color = "grey") +
  stat_summary(fun=mean, colour="firebrick", geom="point") +
  geom_text(data = mean_corr, aes(label = round(metric, 2), x = metric + 0.1), color = "firebrick") +
  labs(x = "AUROC", y = element_blank())

mean_corr = corr_dt2[method!="EVE", mean(metric, na.rm = T), by = .(method)]
colnames(mean_corr)[2] = "metric"
p4 = ggplot(corr_dt2[method!="EVE"], aes(y = method, x = metric)) +
  theme_nature() +
  geom_violin(fill = "lightblue", color = NA) +
  scale_x_continuous(expand = c(0, 0.1)) + 
  theme(panel.grid = element_blank()) +
  geom_jitter(color = "grey") +
  stat_summary(fun=mean, colour="firebrick", geom="point") +
  geom_text(data = mean_corr, aes(label = round(metric, 2), x = metric + 0.1), color = "firebrick") +
  labs(x = "AUROC", y = element_blank())
  
# MCC
corr_dt = fread("MCC_summary.txt")
corr_dt[, geneset:="other"]
corr_dt[complete.cases(corr_dt), geneset:="EVE gene"]
colnames(corr_dt)[which(colnames(corr_dt)=="model_damage")]="MisFit_D"
colnames(corr_dt)[which(colnames(corr_dt)=="model_selection")]="MisFit_S"
colnames(corr_dt)[which(colnames(corr_dt)=="CADD_phred")]="CADD"

corr_dt2 = melt(corr_dt,
                measure.vars = colnames(corr_dt)[is.element(colnames(corr_dt), scores)], 
                id.vars = c('Symbol', 'geneset'),
                variable.name = "method",
                variable.factor = T,
                value.name = "metric"
)
corr_dt2[, method:=factor(method, levels = scores)]

mean_corr = corr_dt2[geneset == "EVE gene", mean(metric, na.rm = T), by = .(method)]
colnames(mean_corr)[2] = "metric"
p5 = ggplot(corr_dt2[geneset == "EVE gene"], aes(y = method, x = metric)) +
  theme_nature() +
  geom_violin(fill = "lightblue", color = NA) +
  scale_x_continuous(expand = c(0, 0.1)) + 
  theme(panel.grid = element_blank()) +
  geom_jitter(color = "grey") +
  stat_summary(fun=mean, colour="firebrick", geom="point") +
  geom_text(data = mean_corr, aes(label = round(metric, 2), x = metric + 0.1), color = "firebrick") +
  labs(x = "MCC", y = element_blank())

mean_corr = corr_dt2[method!="EVE", mean(metric, na.rm = T), by = .(method)]
colnames(mean_corr)[2] = "metric"
p6 = ggplot(corr_dt2[method!="EVE"], aes(y = method, x = metric)) +
  theme_nature() +
  geom_violin(fill = "lightblue", color = NA) +
  scale_x_continuous(expand = c(0, 0.1)) + 
  theme(panel.grid = element_blank()) +
  geom_jitter(color = "grey") +
  stat_summary(fun=mean, colour="firebrick", geom="point") +
  geom_text(data = mean_corr, aes(label = round(metric, 2), x = metric + 0.1), color = "firebrick") +
  labs(x = "MCC", y = element_blank())

# distribution of sensitivity
data_all = fread("DMS_combined_label.txt.gz")
data_all = merge(data_all, corr_dt[,.(Symbol, geneset)])
colnames(data_all)[which(colnames(data_all)=="model_damage")]="MisFit_D"
colnames(data_all)[which(colnames(data_all)=="model_selection")]="MisFit_S"
colnames(data_all)[which(colnames(data_all)=="CADD_phred")]="CADD"
data_all[, target:=factor(target)]
eg_global_recall = 0.5

recall_dt = data.table()
for (global_recall in seq(0.2, 0.8, 0.1)) {
  for (score in scores) {
    roc_all = data.table(roc_curve(data_all[geneset=="EVE gene"], target, score, na_rm = T, event_level = "second"))
    thresh = roc_all[sensitivity - global_recall < 0, min(.threshold)]
    for (genename in unique(data_all[geneset=="EVE gene"]$Symbol)) {
      data = data_all[Symbol==genename]
      data[, pred:=get(score)]
      data[, pred:=factor(as.integer(pred>thresh), levels = c(0, 1))]
      gene_recall = recall(data, target, pred, na_rm = T, event_level = "second")$.estimate
      recall_dt = rbind(recall_dt, data.table(Symbol = genename, global_recall = global_recall, gene_recall = gene_recall, method = score))
    }
  }
}
recall_dt[, method:=factor(method, levels = scores)]
sd_dt = recall_dt[, sd(gene_recall, na.rm = T), by = .(method, global_recall)]
sd_dt[, method:=factor(method, levels = rev(scores))]
colnames(sd_dt)[3] = "metric"
mean_corr = recall_dt[, mean(gene_recall, na.rm = T), by = .(method)]
colnames(mean_corr)[2] = "metric"

p7 = ggplot(recall_dt[global_recall == eg_global_recall], aes(y = method, x = gene_recall)) +
  theme_nature() +
  geom_violin(fill = "bisque", color = NA) +
  scale_x_continuous(expand = c(0, 0.1)) + 
  theme(panel.grid = element_blank()) +
  geom_jitter(color = "grey") +
  geom_vline(xintercept = eg_global_recall, linetype = "dashed") +
  geom_text(data = sd_dt[global_recall == eg_global_recall], aes(label = round(metric, 2)), x = 0.1, color = "blue") +
#  geom_point(data = mean_corr, aes(x = metric), color = "firebrick") +
#  geom_text(data = mean_corr, aes(label = round(metric, 2), x = metric + 0.1), color = "firebrick") +
  labs(x = "sensitivity", y = element_blank()) +
  geom_text_npc(npcx = "left", npcy = "top", label = "std dev", color = "blue", vjust = -1)

p8 = ggplot(sd_dt, aes(x = global_recall, y = metric, color = method)) + 
  theme_nature() + 
  scale_color_manual(values = distinct_colors, drop = F) +
  geom_line() +
  xlab("global sensitivity") +
  ylab("sd of sensitivity")

recall_dt = data.table()
for (global_recall in seq(0.2, 0.8, 0.1)) {
  for (score in setdiff(scores, "EVE")) {
    roc_all = data.table(roc_curve(data_all, target, score, na_rm = T, event_level = "second"))
    thresh = roc_all[sensitivity - global_recall < 0, min(.threshold)]
    for (genename in unique(data_all$Symbol)) {
      data = data_all[Symbol==genename]
      data[, pred:=get(score)]
      data[, pred:=factor(as.integer(pred>thresh), levels = c(0, 1))]
      gene_recall = recall(data, target, pred, na_rm = T, event_level = "second")$.estimate
      recall_dt = rbind(recall_dt, data.table(Symbol = genename, global_recall = global_recall, gene_recall = gene_recall, method = score))
    }
  }
}
recall_dt[, method:=factor(method, levels = scores)]
sd_dt = recall_dt[, sd(gene_recall, na.rm = T), by = .(method, global_recall)]
sd_dt[, method:=factor(method, levels = rev(scores))]
colnames(sd_dt)[3] = "metric"
mean_corr = recall_dt[, mean(gene_recall, na.rm = T), by = .(method)]
colnames(mean_corr)[2] = "metric"

p9 = ggplot(recall_dt[global_recall == eg_global_recall], aes(y = method, x = gene_recall)) +
  theme_nature() +
  geom_violin(fill = "bisque", color = NA) +
  scale_x_continuous(expand = c(0, 0.1)) + 
  theme(panel.grid = element_blank()) +
  geom_jitter(color = "grey") +
  geom_vline(xintercept = eg_global_recall, linetype = "dashed") +
  geom_text(data = sd_dt[global_recall == eg_global_recall], aes(label = round(metric, 2)), x = 0.1, color = "blue") +
#  geom_point(data = mean_corr, aes(x = metric), color = "firebrick") +
#  geom_text(data = mean_corr, aes(label = round(metric, 2), x = metric + 0.1), color = "firebrick") +
  labs(x = "sensitivity", y = element_blank()) +
  geom_text_npc(npcx = "left", npcy = "top", label = "std dev", color = "blue", vjust = -1)

p10 = ggplot(sd_dt, aes(x = global_recall, y = metric, color = method)) + 
  theme_nature() + 
  scale_color_manual(values = distinct_colors, drop = F) +
  geom_line() +
  xlab("global sensitivity") +
  ylab("sd of sensitivity")



arrange_nature(p1, p3, p5, p7, nrow = 1, ncol = 4)
save_nature("figure/summary_EVEgene.pdf", hw_ratio = 1/2, width = "double")

arrange_nature(p2, p4, p6, p9, nrow = 1, ncol = 4)
save_nature("figure/summary_allgene.pdf", hw_ratio = 1/2, width = "double")

arrange_nature(p10, p8, nrow = 1, ncol = 2, common.legend = T, legend = "right")
save_nature("figure/sensitivity_sd.pdf", hw_ratio = 1/2.5, width = "single")


