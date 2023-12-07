library(ggplot2)
library(data.table)
library(stringr)
library(ggrepel)
library(pROC)
library(PRROC)
library(ggpubr)
library(yardstick)
source("../../../settings/plot_settings.R")
scores = c("MisFit_D", "MisFit_S", 'ESM-2', 'ESM-1b', 'AlphaMissense', 'EVE', 'PrimateAI', 'gMVP', 'CADD', 'REVEL', 'MPC')
method_dt = data.table(method = scores)
method_dt[, method_type:="unsupervised"]
method_dt[is.element(method, c("CADD", "gMVP", "REVEL", "MPC")), method_type:="supervised"]

corr_dt = fread("correlation_summary.txt")
#corr_dt = corr_dt[DMS!="DDX3X"]
corr_dt[, geneset:="other"]
corr_dt[!is.na(EVE), geneset:="EVE gene"]
colnames(corr_dt)[which(colnames(corr_dt)=="model_damage")]="MisFit_D"
colnames(corr_dt)[which(colnames(corr_dt)=="model_selection")]="MisFit_S"
colnames(corr_dt)[which(colnames(corr_dt)=="CADD_phred")]="CADD"

corr_dt[is.na(corr_dt), ] = 0

corr_dt2 = melt(corr_dt,
                measure.vars = colnames(corr_dt)[is.element(colnames(corr_dt), scores)], 
                id.vars = c('DMS', 'Symbol', 'num_AA_change', 'geneset'),
                variable.name = "method",
                variable.factor = T,
                value.name = "metric"
)
corr_dt2[, method:=factor(method, levels = scores)]

mean_corr = corr_dt2[, mean(metric, na.rm = T), by = .(method, geneset)]
colnames(mean_corr)[3] = "metric"
mean_corr2 = corr_dt2[, mean(metric, na.rm = T), by = method]
colnames(mean_corr2)[2] = "metric"
mean_corr2[, geneset:="all"]
mean_corr = rbind(mean_corr, mean_corr2[,.(method, geneset, metric)])
mean_corr[, method:=factor(method, levels = scores)]

#mean_corr[is.na(metric), metric:=0]
p1 = ggplot(mean_corr[geneset == "all"], aes(x = reorder(method, metric), y = metric, fill = is.element(method, c("MisFit_D", "MisFit_S")))) + 
  geom_bar(stat = "identity") + 
  coord_flip() +
  theme_bw() +
  theme(legend.position = "none") +
  scale_fill_manual(values = c("grey70", "firebrick")) +
  labs(title = "all", y = expression(mean~spearman~rho), x = "")
p2 = ggplot(mean_corr[geneset == "EVE gene"], aes(x = reorder(method, metric), y = metric, fill = is.element(method, c("MisFit_D", "MisFit_S")))) + 
  geom_bar(stat = "identity") + 
  coord_flip() +
  theme_bw() +
  theme(legend.position = "none") +
  scale_fill_manual(values = c("grey70", "firebrick")) +
  labs(title = "EVE genes", y = expression(mean~spearman~rho), x = "")
p3 = ggplot(mean_corr[geneset == "other"], aes(x = reorder(method, metric), y = metric, fill = is.element(method, c("MisFit_D", "MisFit_S")))) + 
  geom_bar(stat = "identity") + 
  coord_flip() +
  theme_bw() +
  theme(legend.position = "none") +
  scale_fill_manual(values = c("grey70", "firebrick")) +
  labs(title = "other", y = expression(mean~spearman~rho), x = "")
arrange_nature(p1, p2, p3, ncol = 3)
save_nature("figure/spearman_bar.pdf", hw_ratio = 1/3, width_ratio = 1)

corr_dt = fread("AUC_combined_summary.txt")
corr_dt[, geneset:="other"]
corr_dt[!is.na(EVE), geneset:="EVE gene"]
colnames(corr_dt)[which(colnames(corr_dt)=="model_damage")]="MisFit_D"
colnames(corr_dt)[which(colnames(corr_dt)=="model_selection")]="MisFit_S"
colnames(corr_dt)[which(colnames(corr_dt)=="CADD_phred")]="CADD"
corr_dt[is.na(corr_dt), ] = 0

corr_dt2 = melt(corr_dt,
                measure.vars = colnames(corr_dt)[is.element(colnames(corr_dt), scores)], 
                id.vars = c('Symbol', 'num_AA_change', 'geneset'),
                variable.name = "method",
                variable.factor = T,
                value.name = "metric"
)
corr_dt2 = merge(corr_dt2, method_dt, by = "method")
corr_dt2[, method:=factor(method, levels = scores)]

mean_auc = corr_dt2[, mean(metric, na.rm = T), by = .(method, geneset)]
colnames(mean_auc)[3] = "metric"
mean_auc2 = corr_dt2[, mean(metric, na.rm = T), by = method]
colnames(mean_auc2)[2] = "metric"
mean_auc2[, geneset:="all"]
mean_auc = rbind(mean_auc, mean_auc2[,.(method, geneset, metric)])
mean_auc[, method:=factor(method, levels = scores)]
#mean_auc[is.na(metric), metric:=0]

p4 = ggplot(mean_auc[geneset == "all"], aes(x = reorder(method, metric), y = metric, fill = is.element(method, c("MisFit_D", "MisFit_S")))) + 
  geom_bar(stat = "identity") + 
  coord_flip() +
  theme_bw() +
  theme(legend.position = "none") +
  scale_fill_manual(values = c("grey70", "firebrick")) +
  labs(title = "all", y = "mean AUROC", x = "")
p5 = ggplot(mean_auc[geneset == "EVE gene"], aes(x = reorder(method, metric), y = metric, fill = is.element(method, c("MisFit_D", "MisFit_S")))) + 
  geom_bar(stat = "identity") + 
  coord_flip() +
  theme_bw() +
  theme(legend.position = "none") +
  scale_fill_manual(values = c("grey70", "firebrick")) +
  labs(title = "EVE genes", y = "mean AUROC", x = "")
p6 = ggplot(mean_auc[geneset == "other"], aes(x = reorder(method, metric), y = metric, fill = is.element(method, c("MisFit_D", "MisFit_S")))) + 
  geom_bar(stat = "identity") + 
  coord_flip() +
  theme_bw() +
  theme(legend.position = "none") +
  scale_fill_manual(values = c("grey70", "firebrick")) +
  labs(title = "other", y = "mean AUROC", x = "")
arrange_nature(p4, p5, p6, ncol = 3)
save_nature("figure/AUC_bar.pdf", hw_ratio = 1/3, width_ratio = 1)

data_all = fread("DMS_combined_label.txt.gz")
data_all = merge(data_all, corr_dt[,.(Symbol, geneset)])
colnames(data_all)[which(colnames(data_all)=="model_damage")]="MisFit_D"
colnames(data_all)[which(colnames(data_all)=="model_selection")]="MisFit_S"
colnames(data_all)[which(colnames(data_all)=="CADD_phred")]="CADD"
data_all[is.na(data_all)] = 0
corr_dt3 = data.table()
for (method in scores) {
  combined = roc(data_all[, target], 
                 data_all[, get(method)], direction = "<")
  AUC = as.numeric(auc(combined))
  corr_dt3 = rbind(corr_dt3, data.table(method = method, geneset = "all", metric = AUC))
}

for (method in scores) {
  combined = roc(data_all[geneset=="EVE gene", target], 
                 data_all[geneset=="EVE gene", get(method)], direction = "<")
  AUC = as.numeric(auc(combined))
  corr_dt3 = rbind(corr_dt3, data.table(method = method, geneset = "EVE gene", metric = AUC))
}

for (method in scores) {
    combined = roc(data_all[geneset=="other", target], 
                   data_all[geneset=="other", get(method)], direction = "<")
    AUC = as.numeric(auc(combined))
  corr_dt3 = rbind(corr_dt3, data.table(method = method, geneset = "other", metric = AUC))
}

p7 = ggplot(corr_dt3[geneset == "all"], aes(x = reorder(method, metric), y = metric, fill = is.element(method, c("MisFit_D", "MisFit_S")))) + 
  geom_bar(stat = "identity") + 
  coord_flip() +
  theme_bw() +
  theme(legend.position = "none") +
  scale_fill_manual(values = c("grey70", "firebrick")) +
  labs(title = "all", y = "combined AUROC", x = "")
p8 = ggplot(corr_dt3[geneset == "EVE gene"], aes(x = reorder(method, metric), y = metric, fill = is.element(method, c("MisFit_D", "MisFit_S")))) + 
  geom_bar(stat = "identity") + 
  coord_flip() +
  theme_bw() +
  theme(legend.position = "none") +
  scale_fill_manual(values = c("grey70", "firebrick")) +
  labs(title = "EVE genes", y = "combined AUROC", x = "")
p9 = ggplot(corr_dt3[geneset == "other"], aes(x = reorder(method, metric), y = metric, fill = is.element(method, c("MisFit_D", "MisFit_S")))) + 
  geom_bar(stat = "identity") + 
  coord_flip() +
  theme_bw() +
  theme(legend.position = "none") +
  scale_fill_manual(values = c("grey70", "firebrick")) +
  labs(title = "other", y = "combined AUROC", x = "")
arrange_nature(p7, p8, p9, ncol = 3)
save_nature("figure/combined_AUC_bar.pdf", hw_ratio = 1/3, width_ratio = 1)

p10 = ggplot(mean_corr[(geneset == "all")&(method!="EVE")], aes(x = reorder(method, metric), y = metric, fill = is.element(method, c("MisFit_D", "MisFit_S")))) + 
  geom_bar(stat = "identity") + 
  coord_flip() +
  theme_bw() +
  theme(legend.position = "none") +
  scale_fill_manual(values = c("grey70", "firebrick")) +
  labs(y = expression(mean~spearman~rho), x = "")

p11 = ggplot(mean_auc[(geneset == "all")&(method!="EVE")], aes(x = reorder(method, metric), y = metric, fill = is.element(method, c("MisFit_D", "MisFit_S")))) + 
  geom_bar(stat = "identity") + 
  coord_flip() +
  theme_bw() +
  theme(legend.position = "none") +
  scale_fill_manual(values = c("grey70", "firebrick")) +
  labs(y = "mean AUROC", x = "")

p12 = ggplot(corr_dt3[(geneset == "all")&(method!="EVE")], aes(x = reorder(method, metric), y = metric, fill = is.element(method, c("MisFit_D", "MisFit_S")))) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_bw() +
  theme(legend.position = "none") +
  scale_fill_manual(values = c("grey70", "firebrick")) +
  labs(y = "combined AUROC", x = "")

auc_dt = merge(mean_auc, corr_dt3, by = c("method", "geneset"))
auc_dt = merge(auc_dt, method_dt, by = "method")
p13 = ggplot(auc_dt[(geneset == "all")&(method!="EVE")], aes(x = metric.x, y = metric.y)) + 
  geom_text_repel(aes(label = method, color = is.element(method, c("MisFit_D", "MisFit_S"))), box.padding = 0.8) +
  geom_point(size = 3, aes(shape = method_type, color = is.element(method, c("MisFit_D", "MisFit_S"))), stroke = 1) +
  theme_nature() + 
  scale_color_manual(values = c("grey50", "firebrick"), guide = "none") +
  scale_shape_manual(name = "", values = c(16, 1)) +
  scale_x_continuous(name = "mean DMS AUROC", breaks = seq(0.5, 1, 0.1), minor_breaks = c(), limits = c(0.45, 0.9)) +
  scale_y_continuous(name = "combined DMS AUROC", breaks = seq(0.5, 1, 0.1), minor_breaks = c(), limits = c(0.45, 0.9)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  theme(legend.position = "bottom") +
  coord_fixed()


mcc_dt = fread("MCC_summary.txt")
mcc_dt = merge(mcc_dt, corr_dt[,.(Symbol, geneset)])
colnames(mcc_dt)[which(colnames(mcc_dt)=="model_damage")]="MisFit_D"
colnames(mcc_dt)[which(colnames(mcc_dt)=="model_selection")]="MisFit_S"
colnames(mcc_dt)[which(colnames(mcc_dt)=="CADD_phred")]="CADD"
mcc_dt[is.na(mcc_dt)] = 0
mcc_dt2 = melt(mcc_dt,
                measure.vars = colnames(mcc_dt)[is.element(colnames(mcc_dt), scores)], 
                id.vars = c('Symbol', 'geneset'),
                variable.name = "method",
                variable.factor = T,
                value.name = "metric"
)
mcc_dt2[, method:=factor(method, levels = scores)]


p14 = add_summary(ggplot(mcc_dt2, aes(x = reorder(method, -metric), y = metric)) +
              theme_nature() +
              theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
              labs(x = "", y = "MCC", title = "all") +
                geom_violin() +
              geom_jitter(width = 0.1, color = "grey70"),
            fun = "mean",
            color = "black", size = 0.5)
p15 = add_summary(ggplot(mcc_dt2[geneset=="EVE gene"], aes(x = reorder(method, -metric), y = metric)) + 
                    theme_nature() +
                    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
                    labs(x = "", y = "MCC", title = "EVE genes") +
                    geom_violin() +
                    geom_jitter(width = 0.1, color = "grey70"),
                  fun = "mean", 
                  color = "black",
                  size = 0.5)
p16 = add_summary(ggplot(mcc_dt2[geneset=="other"], aes(x = reorder(method, -metric), y = metric)) + 
                    theme_nature() +
                    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
                    labs(x = "", y = "MCC", title = "other") +
                    geom_violin() +
                    geom_jitter(width = 0.1, color = "grey70"),
                  fun = "mean", 
                  color = "black",
                  size = 0.5)
arrange_nature(p14, p15, p16, nrow = 3)
save_nature("figure/MCC_distri.pdf", hw_ratio = 1)

p17 = add_summary(ggplot(mcc_dt2[method!="EVE"], aes(x = reorder(method, -metric), y = metric)) + 
                    theme_nature() +
                    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
                    labs(x = "", y = "MCC") +
                    geom_violin() +
                    geom_jitter(width = 0.1, color = "grey70"),
                  fun = "mean", 
                  color = "black",
                  size = 0.5)

data_all[, target:=factor(target)]

recalls = c(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
recall_sd_all = data.table()
for (score in scores) {
  recall_dt = data.table()
  roc_all = data.table(roc_curve(data_all, target, score, na_rm = T, event_level = "second"))
  for (rec in recalls) {
    for (genename in unique(data_all$Symbol)) {
      thresh = roc_all[sensitivity - rec < 0, min(.threshold)]
      data = data_all[Symbol==genename]
      data[, pred:=get(score)]
      data[, pred:=factor(as.integer(pred>thresh), levels = c(0, 1))]
      recall = recall(data, target, pred, na_rm = T, event_level = "second")$.estimate
      recall_dt = rbind(recall_dt, data.table(Symbol = genename, global_recall = rec, gene_recall = recall))
    }
  }
  recall_sd = recall_dt[, sd(gene_recall, na.rm = T), by = global_recall]
  colnames(recall_sd) = c("global_sensitivity", "gene_sensitivity_sd")
  recall_sd[, method:=score]
  recall_sd_all = rbind(recall_sd_all, recall_sd)
}
recall_sd_all[, method:=factor(method, levels = scores)]
p18 = ggplot(recall_sd_all[method!="EVE"], aes(x = global_sensitivity, y = gene_sensitivity_sd, color = method)) + 
  theme_nature() + 
  #  scale_color_brewer(palette = "Spectral") +
  scale_color_manual(values = distinct_colors) +
  geom_line() +
  xlab("global sensitivity") +
  ylab("sd of sensitivity")

specs = c(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
spec_sd_all = data.table()

for (score in scores) {
  spec_dt = data.table()
  roc_all = data.table(roc_curve(data_all, target, score, na_rm = T, event_level = "second"))
  for (spec in specs) {
    for (genename in unique(data_all$Symbol)) {
      thresh = roc_all[specificity - spec < 0, max(.threshold)]
      data = data_all[Symbol==genename]
      data[, pred:=get(score)]
      data[, pred:=factor(as.integer(pred>thresh), levels = c(0, 1))]
      specificity = spec(data, target, pred, na_rm = T, event_level = "second")$.estimate
      spec_dt = rbind(spec_dt, data.table(Symbol = genename, global_spec = spec, gene_spec = specificity))
    }
  }
  spec_sd = spec_dt[, sd(gene_spec, na.rm = T), by = global_spec]
  colnames(spec_sd) = c("global_specificity", "gene_specificity_sd")
  spec_sd[, method:=score]
  spec_sd_all = rbind(spec_sd_all, spec_sd)
}


spec_sd_all[, method:=factor(method, levels = scores)]
p19 = ggplot(spec_sd_all[method!="EVE"], aes(x = global_specificity, y = gene_specificity_sd, color = method)) + 
  theme_nature() + 
  #  scale_color_brewer(palette = "Spectral") +
  scale_color_manual(values = distinct_colors) +
  geom_line() +
  xlab("global specificity") +
  ylab("sd of specificity")

arrange_nature(arrange_nature(p10, p11, p12, p13, ncol = 4, widths = c(1,1,1,1.5), labels = c("a", "b", "c", "d")), 
               arrange_nature(p17, p18, ncol = 2, widths = c(1.2, 1), labels = c("e", "f")),
               nrow = 2, labels = c("", ""),
               heights = c(1, 1))
save_nature("figure/summary_noEVE.pdf", hw_ratio = 2/3, width = "double")



