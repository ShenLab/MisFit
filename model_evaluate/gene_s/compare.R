library(data.table)
library(ggplot2)
library(ggpubr)
library(scales)
library(stringr)
library(PRROC)
library(pROC)

source("../../../settings/plot_settings.R")

sigmoid = function(x) {
  return (1 / (1+exp(-x)))
}

df = fread("geneset_lof_s_combined.txt")
colnames(df)[which(colnames(df)=="logit_s_mean")] = "logit_s_mean_ptv"
colnames(df)[which(colnames(df)=="logit_s_sd")] = "logit_s_sd_ptv"
df[, MisFit_sgene_ptv:=pmax(s_median, 1e-4)]

df2 = fread("geneset_mis_s.txt")
colnames(df2)[which(colnames(df2)=="logit_s_mean")] = "logit_s_mean_mis"
colnames(df2)[which(colnames(df2)=="logit_s_sd")] = "logit_s_sd_mis"
df2[, MisFit_sgene_mis:=sigmoid(logit_s_mean_mis)]

df3 = merge(df, df2, sort = F)
df4 = df3[, .(UniprotID, GeneID, TranscriptID, Symbol, ProteinID, UniprotEntryName, Strand, Chrom, Length,
              MisFit_sgene_mis, MisFit_sgene_ptv, logit_s_mean_mis, logit_s_sd_mis, logit_s_mean_ptv, logit_s_sd_ptv)]
fwrite(df4, "geneset_s_gene.txt", sep = "\t")


p1 = ggplot(df3, aes(x = MisFit_sgene_ptv, y = s_mean_Weghorn)) + 
  theme_nature() + 
  stat_density_2d(aes(fill = ..density..), geom = "raster", contour = FALSE) +
  scale_fill_gradient(low = "white", high = "red") +
  scale_x_continuous(name = "MisFit S PTV", limits = c(10^-4, 1), trans = "log10", labels = trans_format("log10", math_format(10^.x))) +
  scale_y_continuous(name = expression(s[het]~Weghorn2019), limits = c(10^-4, 1), trans = "log10", labels = trans_format("log10", math_format(10^.x)))+
  theme(legend.position = "none") +
  coord_fixed() +
  stat_cor(data = df, aes(x = log(s_median+1), y = log(s_mean_Weghorn+1), label = ..r.label..), method = "pearson")

p2 = ggplot(df3, aes(x = MisFit_sgene_ptv, y = s_mode_Agarwal)) + 
  theme_nature() + 
  stat_density_2d(aes(fill = ..density..), geom = "raster", contour = FALSE) +
  scale_fill_gradient(low = "white", high = "red") +
  scale_x_continuous(name = "MisFit S PTV", limits = c(10^-4, 1), trans = "log10", labels = trans_format("log10", math_format(10^.x))) +
  scale_y_continuous(name = expression(s[het]~Agarwal2023), limits = c(10^-4, 1), trans = "log10", labels = trans_format("log10", math_format(10^.x)))+
  theme(legend.position = "none") +
  coord_fixed() +
  stat_cor(data = df, aes(x = log(s_median+1), y = log(s_mode_Agarwal+1), label = ..r.label..), method = "pearson")

p3 = ggplot(df3, aes(x = MisFit_sgene_ptv, y = s_mean_Zeng)) + 
  theme_nature() + 
  stat_density_2d(aes(fill = ..density..), geom = "raster", contour = FALSE) +
  scale_fill_gradient(low = "white", high = "red") +
  scale_x_continuous(name = "MisFit S PTV", limits = c(10^-4, 1), trans = "log10", labels = trans_format("log10", math_format(10^.x))) +
  scale_y_continuous(name = expression(s[het]~Zeng2023), limits = c(10^-4, 1), trans = "log10", labels = trans_format("log10", math_format(10^.x)))+
  theme(legend.position = "none") +
  coord_fixed() +
  stat_cor(data = df, aes(x = log(s_median+1), y = log(s_mean_Zeng+1), label = ..r.label..), method = "pearson")

arrange_nature(p1, p2, p3, ncol = 3)
save_nature("figure/lof_compare.pdf", hw_ratio = 1.2/3, width_ratio = 1)

p4 = ggplot(df3, aes(x = MisFit_sgene_mis, y = mis_z)) + 
  theme_nature() + 
  geom_point(aes(color = log10(Length)), alpha = 0.5) +
  scale_x_continuous(name = expression(MisFit~S[gene]), trans = "log10", labels = trans_format("log10", math_format(10^.x))) +
  scale_y_continuous(name = "gnomAD missense z", limits = c(-10, 10)) +
  scale_color_distiller(name = expression(length~(10^3)), breaks = c(2,3,4), labels = c(0.1,1,10), palette="YlGnBu", direction = 0) +
  stat_cor(aes(label = ..r.label.. ), method = "spearman", cor.coef.name = "rho") +
  theme(aspect.ratio = 1)

p5 = ggplot(df3, aes(x = MisFit_sgene_mis, y = oe_mis)) + 
  theme_nature() + 
  geom_point(aes(color = log10(Length)), alpha = 0.5) +
  scale_x_continuous(name = expression(MisFit~S[gene]), trans = "log10", labels = trans_format("log10", math_format(10^.x))) +
  scale_y_continuous(name = "gnomAD missense o/e", limits = c(0, 3)) +
  scale_color_distiller(name = expression(length~(10^3)), breaks = c(2,3,4), labels = c(0.1,1,10), palette="YlGnBu", direction = 0) +
  stat_cor(aes(label = ..r.label..), method = "spearman", cor.coef.name = "rho") +
  coord_fixed() +
  theme(aspect.ratio = 1)
  
arrange_nature(p4, p5, common.legend = T, legend = "right")
save_nature("figure/s_mis.pdf", hw_ratio = 1/2)
save_nature("figure/s_mis.png", hw_ratio = 1/2)


