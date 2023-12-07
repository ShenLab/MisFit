library(data.table)
library(ggplot2)
library(ggrepel)
source("../../../settings/plot_settings.R")

logit = function(x) {
  return(log(x/(1-x)))
}

scores = c("model0", "model1", "model2", "MisFit_S")
# ASD
syn0 = 1779
syn1 = 5264
dt = fread("ASD_mis_hg19_merged.txt.gz")

colnames(dt)[which(colnames(dt)=="model_selection")]="MisFit_S"

dt0 = dt[Pheno=="Unaffected"]

threshs = c(0.01, 0.02, 0.05, 0.1, 0.2)
summary_dt = data.table()
for (score in scores) {
  dt_sub = dt
  dt_sub[, rankscore := get(score)]
  dt_sub[, rankscore := frank(rankscore, na.last = F)]
  dt_sub[, rankscore := rankscore / nrow(dt_sub)]
  
  for (thresh in threshs) {
    m0 = nrow(dt_sub[(rankscore>=(1-thresh))&(Pheno=="Unaffected")] )
    m1 = nrow(dt_sub[(rankscore>=(1-thresh))&(Pheno=="Affected")])
    m0_norm = m0/syn0*syn1
    mpos = m1 - m0_norm
    summary_dt = rbind(summary_dt, data.table(enrich = m1/m0_norm, precision = mpos / m1, recall = mpos, thresh = thresh, method = score, annot = NA))
  }
}

m0 = nrow(dt[(Pheno=="Unaffected")] )
m1 = nrow(dt[(Pheno=="Affected")])
m0_norm = m0/syn0*syn1
mpos = m1 - m0_norm
summary_dt = rbind(summary_dt, data.table(enrich = m1/m0_norm, precision = mpos / m1, recall = mpos, thresh = NA, method = NA, annot = NA))
summary_dt[, method:=factor(method, levels = c("model0", "model1", "model2", "MisFit_S"), 
                            labels = c("model 0", "model 1", "model 2", "MisFit_S"))]
p1 = ggplot(summary_dt[!is.na(method)], aes(x = thresh * 100, y = enrich, color = method, group = method)) + 
  theme_nature() +
  geom_path() +
  scale_x_continuous("top percentile", breaks = threshs * 100, minor_breaks = NULL) +
  scale_y_continuous("enrichment ratio", trans = "log2") +
#  geom_point(data = subset(summary_dt, !is.na(annot)), aes(x = thresh * 100, y = enrich, color = method)) +
#  geom_text_repel(data = subset(summary_dt, !is.na(annot)), aes(x = thresh * 100, y = enrich, label = annot), nudge_x = 0.5, nudge_y = 0.1) +
  geom_hline(data = summary_dt[is.na(thresh)], aes(yintercept = enrich)) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  ggtitle("autism")

p2 = ggplot(summary_dt[!is.na(method)], aes(x = recall, y = precision, color = method, group = method)) + 
  theme_nature() +
  geom_path() +
  scale_x_continuous("# estimated risk variants") +
  scale_y_continuous("estimated precision") +
#  geom_point(data = subset(summary_dt, !is.na(annot)), aes(x = recall, y = precision, color = method)) +
#  geom_text_repel(data = subset(summary_dt, !is.na(annot)), aes(x = recall, y = precision, label = annot), box.padding = 0.5) +
  ggtitle("autism")


# NDD
syn0 = 1779
syn1 = 9203
dt = fread("NDD_mis_hg19_merged.txt.gz")

colnames(dt)[which(colnames(dt)=="model_selection")]="MisFit_S"

dt = rbind(dt, dt0)

threshs = c(0.01, 0.02, 0.05, 0.1, 0.2)
summary_dt = data.table()
for (score in scores) {
  dt_sub = dt
  dt_sub[, rankscore := get(score)]
  dt_sub[, rankscore := frank(rankscore, na.last = F)]
  dt_sub[, rankscore := rankscore / nrow(dt_sub)]
  
  for (thresh in threshs) {
    m0 = nrow(dt_sub[(rankscore>=(1-thresh))&(Pheno=="Unaffected")] )
    m1 = nrow(dt_sub[(rankscore>=(1-thresh))&(Pheno=="Affected")])
    m0_norm = m0/syn0*syn1
    mpos = m1 - m0_norm
    summary_dt = rbind(summary_dt, data.table(enrich = m1/m0_norm, precision = mpos / m1, recall = mpos, thresh = thresh, method = score, annot = NA))
  }
  
}

m0 = nrow(dt[(Pheno=="Unaffected")] )
m1 = nrow(dt[(Pheno=="Affected")])
m0_norm = m0/syn0*syn1
mpos = m1 - m0_norm
summary_dt = rbind(summary_dt, data.table(enrich = m1/m0_norm, precision = mpos / m1, recall = mpos, thresh = NA, method = NA, annot = NA))
summary_dt[, method:=factor(method, levels = c("model0", "model1", "model2", "MisFit_S"), 
                            labels = c("model 0", "model 1", "model 2", "MisFit_S"))]
p3 = ggplot(summary_dt[!is.na(method)], aes(x = thresh * 100, y = enrich, color = method, group = method)) + 
  theme_nature() +
  geom_path() +
  scale_x_continuous("top percentile", breaks = threshs * 100, minor_breaks = NULL) +
  scale_y_continuous("enrichment ratio", trans = "log2") +
#  geom_point(data = subset(summary_dt, !is.na(annot)), aes(x = thresh * 100, y = enrich, color = method)) +
#  geom_text_repel(data = subset(summary_dt, !is.na(annot)), aes(x = thresh * 100, y = enrich, label = annot), nudge_x = 0.5, nudge_y = 0.1) +
  geom_hline(data = summary_dt[is.na(thresh)], aes(yintercept = enrich)) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  ggtitle("NDD")

p4 = ggplot(summary_dt[!is.na(method)], aes(x = recall, y = precision, color = method, group = method)) + 
  theme_nature() +
  geom_path() +
  scale_x_continuous("# estimated risk variants") +
  scale_y_continuous("estimated precision") +
#  geom_point(data = subset(summary_dt, !is.na(annot)), aes(x = recall, y = precision, color = method)) +
#  geom_text_repel(data = subset(summary_dt, !is.na(annot)), aes(x = recall, y = precision, label = annot), box.padding = 0.5) +
  ggtitle("NDD")

arrange_nature(p1,p3, common.legend = T, legend = "right")
save_nature("figure/DNV_enrichment_basic.pdf",  hw_ratio = 1/2)

arrange_nature(p2,p4, common.legend = T, legend = "right")
save_nature("figure/DNV_PR_basic.pdf",  hw_ratio = 1/2)
