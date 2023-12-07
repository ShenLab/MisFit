library(data.table)
library(ggplot2)
library(ggrepel)
library(scales)
source("../../../settings/plot_settings.R")

logit = function(x) {
  return(log(x/(1-x)))
}

sigmoid = function(x) {
  return(1/(1+exp(-x)))
}

data_all = data.table()
scores = c("MisFit_D", "MisFit_S")

dt = fread("ASD_mis_hg19_merged.txt.gz")
colnames(dt)[which(colnames(dt)=="model_damage")]="MisFit_D"
colnames(dt)[which(colnames(dt)=="model_selection")]="MisFit_S"
dt[Pheno=="Affected", Pheno:="Autism"]
data_all = rbind(data_all, dt[,.(MisFit_D, MisFit_S, Pheno)])

dt = fread("NDD_mis_hg19_merged.txt.gz")
colnames(dt)[which(colnames(dt)=="model_damage")]="MisFit_D"
colnames(dt)[which(colnames(dt)=="model_selection")]="MisFit_S"
dt[, Pheno:="NDD"]
data_all = rbind(data_all, dt[,.(MisFit_D, MisFit_S, Pheno)])

# dt = fread("CDH_mis_merged.txt.gz")
# colnames(dt)[which(colnames(dt)=="model_damage")]="MisFit_D"
# colnames(dt)[which(colnames(dt)=="model_selection")]="MisFit_S"
# dt[, Pheno:="CDH"]
# data_all = rbind(data_all, dt[,.(MisFit_D, MisFit_S, Pheno)])
# 
# dt = fread("EA_mis_hg19_merged.txt.gz")
# colnames(dt)[which(colnames(dt)=="model_damage")]="MisFit_D"
# colnames(dt)[which(colnames(dt)=="model_selection")]="MisFit_S"
# dt[, Pheno:="EA/TEF"]
# data_all = rbind(data_all, dt[,.(MisFit_D, MisFit_S, Pheno)])

p1 = ggplot(data_all, aes(x = sigmoid(MisFit_S), group = Pheno, color = Pheno)) + geom_density() + theme_nature() +
#  scale_color_manual(name = "", values = c("red", "blue", "lightblue", "orange", "black")) +
  scale_x_continuous(name = "MisFit S",  trans = "log10", labels = trans_format("log10", math_format(10^.x))) +
  labs(title = "missense")

p2 = ggplot(data_all, aes(x = MisFit_D, group = Pheno, color = Pheno)) + geom_density() + theme_nature() +
  scale_x_continuous(name = "MisFit D", breaks = seq(0, 1, 0.2)) +
  labs(title = "missense")

#testp = wilcox.test(dt[Pheno=="Unaffected", MisFit_s], dt[Pheno=="Affected", MisFit_s])$p.value
#ggplot(dt, aes(x = sigmoid(MisFit_s), group = Pheno, color = Pheno)) + geom_density() + theme_nature() +
#  scale_color_manual(name = "", values = c("red", "blue")) +
#  scale_x_continuous(name = "MisFit s",  trans = "log10", labels = trans_format("log10", math_format(10^.x))) +
#  ggtitle(paste0("p = ", signif(testp, 2)))



data_all = c()
dt = fread("ASD_DNV_lof_hg19_merged.txt.gz")
dt[Pheno=="Affected", Pheno:="Autism"]
data_all = rbind(data_all, dt[,.(s_median, s_mean, s_mean_Zeng, s_mean_Weghorn, s_mode_Agarwal, Pheno)])

dt = fread("NDD_DNV_lof_hg19_merged.txt.gz")
dt[, Pheno:="NDD"]
data_all = rbind(data_all, dt[,.(s_median, s_mean, s_mean_Zeng, s_mean_Weghorn, s_mode_Agarwal, Pheno)])

# dt = fread("CDH_DNV_lof_merged.txt.gz")
# dt[, Pheno:="CDH"]
# data_all = rbind(data_all, dt[,.(s_median, s_mean, s_mean_Zeng, s_mean_Weghorn, s_mode_Agarwal, Pheno)])
# 
# dt = fread("EA_DNV_lof_hg19_merged.txt.gz")
# dt[, Pheno:="EA/TEF"]
# data_all = rbind(data_all, dt[,.(s_median, s_mean, s_mean_Zeng, s_mean_Weghorn, s_mode_Agarwal, Pheno)])

p3 = ggplot(data_all, aes(x = s_median, group = Pheno, color = Pheno)) + geom_density() + 
  theme_nature() +
  scale_x_continuous(name = "MisFit S",  trans = "log10", limits =c(1e-4, 1), breaks = c(1e-4, 1e-3, 1e-2, 1e-1), labels = trans_format("log10", math_format(10^.x))) +
  labs(color = "")
  
ld = get_legend(p3)

arrange_nature(p2 + theme(legend.position = "none"), ld,
               p1 + theme(legend.position = "none"), p3 + ggtitle("PTV") + theme(legend.position = "none"),
               labels = c("a", "", "b", "c")
)
save_nature("figure/DNV_hist.pdf", hw_ratio = 1)

# p4 = ggplot(data_all, aes(x = s_mean_Weghorn, group = Pheno, color = Pheno)) + geom_density() + theme_nature() +
#   scale_x_continuous(name = expression(s[het]~Weghorn2019),  trans = "log10", limits = c(1e-4, 1), breaks = c(1e-4, 1e-3, 1e-2, 1e-1), labels = trans_format("log10", math_format(10^.x))) 
# p5 = ggplot(data_all, aes(x = s_mode_Agarwal, group = Pheno, color = Pheno)) + geom_density() + theme_nature() +
#   scale_x_continuous(name = expression(s[het]~Agarwal2023),  trans = "log10", limits = c(1e-4, 1),breaks = c(1e-4, 1e-3, 1e-2, 1e-1), labels = trans_format("log10", math_format(10^.x))) 
# p6 = ggplot(data_all, aes(x = s_mean_Zeng, group = Pheno, color = Pheno)) + geom_density() + theme_nature() +
#   scale_x_continuous(name = expression(s[het]~Zeng2023),  trans = "log10", limits = c(1e-4, 1),breaks = c(1e-4, 1e-3, 1e-2, 1e-1), labels = trans_format("log10", math_format(10^.x))) 
# arrange_nature(p3, p4, p5, p6, common.legend = T)
# save_nature("figure/DNV_lof_hist.pdf", hw_ratio = 1)






