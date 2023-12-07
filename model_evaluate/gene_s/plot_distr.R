library(data.table)
library(ggplot2)
library(scales)
library(ggpubr)
library(cowplot)
library(stringr)
source("../../../settings/plot_settings.R")

df = fread("geneset_s_gene.txt")

sigmoid = function(x) {
  return (1 / (1+exp(-x)))
}

s_min = -9.2
slof = rnorm(length(df$logit_s_mean_ptv), df$logit_s_mean_ptv, df$logit_s_sd_ptv)
slof = pmax(1e-4, sigmoid(slof))

smis = rnorm(length(df$logit_s_mean_mis), df$logit_s_mean_mis, df$logit_s_sd_mis)
degree = sigmoid(rnorm(length(smis), 0.116, 2))
smis = degree * (smis - s_min) + s_min
smis = pmax(1e-4, sigmoid(smis))

sum(smis[!is.na(smis)]>1e-2)/sum(smis[!is.na(smis)]>0)

s_df = rbind(data.table(s=smis, type="missense"), 
             data.table(s=slof, type="PTV"))
s_df = na.omit(s_df)

p1 = ggplot(s_df) + 
  geom_density(aes(x = s, color = type), size = 1) + 
  theme_nature() +
  scale_x_continuous(name = "MisFit S", trans = "log10", labels = trans_format("log10", math_format(10^.x))) +
  scale_color_manual(name = "", values = c("darkblue", "darkred")) +
  theme(legend.position = "top") +
  theme(aspect.ratio=1)

p2 = ggplot(df, aes(x = MisFit_sgene_ptv, y = MisFit_sgene_mis)) +
  theme_nature() +
  geom_point(color = "royalblue", alpha = 0.05) +
  geom_density_2d(bins = 8, color = "black") +
  scale_x_continuous(name = "MisFit S PTV", trans = "log10", limits = c(10^-4, 1), labels = trans_format("log10", math_format(10^.x))) +
  scale_y_continuous(name = expression(MisFit~S[gene]), trans = "log10", limits = c(10^-4, 1), labels = trans_format("log10", math_format(10^.x))) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  theme(legend.position = "top") +
  coord_fixed() +
  stat_cor(data = df, aes(x = log(MisFit_sgene_ptv + 1), y = log(MisFit_sgene_mis + 1), label = ..r.label..), method = "pearson")

  
arrange_nature(p1, p2)
save_nature("figure/s_distr.png", hw_ratio = 1/2, width_ratio = 1)
save_nature("figure/s_distr.pdf", hw_ratio = 1/2, width_ratio = 1)

pathway = fread("protein_mech.csv")
pathway[, Symbol:=Gene]
mech = unique(pathway[Molecular_disease_mechanism!="Unknown", .(Symbol, Molecular_disease_mechanism)])
mech[, type:=Molecular_disease_mechanism]
mech = mech[, .(Symbol, type)]

# cancer = fread("cosmic.csv")
# cancer[, Symbol:=`Gene Symbol`]
# cancer_sub = cancer[str_detect(`Role in Cancer`, "oncogene"), .(Symbol)]
# cancer_sub[, type:="OCG"]
# mech = rbind(mech, cancer_sub)
# cancer_sub = cancer[str_detect(`Role in Cancer`, "TSG"), .(Symbol)]
# cancer_sub[, type:="TSG"]
# mech = rbind(mech, cancer_sub)
all_genetype = unique(mech$type)

df[, ratio := MisFit_sgene_mis / MisFit_sgene_ptv]
mech = merge(df, mech)
fwrite(mech, "known_genes_s.csv")

plot_list = list()
i = 1
for (genetype in all_genetype) {
  p = ggplot() +
    theme_nature() +
    geom_density_2d(data = mech[type==genetype], aes(x = MisFit_sgene_ptv, y = MisFit_sgene_mis), bins = 6, color = "red") +
    geom_density_2d(data = df, aes(x = MisFit_sgene_ptv, y = MisFit_sgene_mis), bins = 8, color = "black") +
    scale_x_continuous(name = "MisFit S PTV", trans = "log10", limits = c(10^-4, 1), labels = trans_format("log10", math_format(10^.x))) +
    scale_y_continuous(name = expression(MisFit~S[gene]), trans = "log10", limits = c(10^-4, 1), labels = trans_format("log10", math_format(10^.x))) +
    coord_fixed() +
    ggtitle(genetype)
    # theme(plot.margin = margin(t = 0,
    #                            r = 0,
    #                            b = 10,
    #                            l = 10)) +
  plot_list[[i]] = p
  i = i + 1
}


arrange_nature(p2, arrange_nature(plotlist = plot_list, nrow = 2, ncol = 2, labels = c("b", "c", "d", "e")), ncol = 2, widths = c(2, 3), labels = c("a", ""))
save_nature("figure/hist_mech.pdf", width_ratio = 1, hw_ratio = 2/3)
save_nature("figure/hist_mech.png", width_ratio = 1, hw_ratio = 2/3)
