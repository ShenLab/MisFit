library(data.table)
library(ggplot2)
library(scales)
library(stringr)
library(viridis)
source("../../../settings/plot_settings.R")
library(rateratio.test)

sigmoid = function(x) {
  return(1/(1+exp(-x)))
}

geneset = fread("combine_covered_list.txt", header = F, col.names = "UniprotID")
geneset_all = fread("geneset_uniprot.txt")
geneset = merge(geneset, geneset_all)$Symbol

both = fread("ASD_SPARK0_missense_merged.txt.gz")
both = both[is.element(Symbol, geneset)&(AC/AN<0.1)&(outlier==F)&(Filter=='')]
asd = fread("ASD_mis_hg19_merged.txt.gz")
asd = asd[is.element(Symbol, geneset)]
both[, s:=sigmoid(model_selection)]
denovo_var_id = asd[, paste(Symbol,Uniprot_position, AA_alt, sep = ":")]
both_var_id = both[, paste(Symbol,Uniprot_position, AA_alt, sep = ":")]

inherited = both[!is.element(both_var_id, denovo_var_id)]

#inherited[AC/AN>1e-4, type:="inherited (AF > 1e-4)"]
#inherited[AC/AN<1e-4, type:="inherited (1e-5 < AF < 1e-4)"]
#inherited[AC/AN<1e-5, type:="inherited (0 < AF < 1e-5)"]
#inherited[AC==0, type:="inherited (AF = 0)"]
inherited[, type:="inherited"]
inherited = inherited[!is.na(type), .(s, AC_Affected, AC_Unaffected, type)]
colnames(inherited)[c(2,3)] = c("Affected", "Unaffected")
inherited = melt(data = inherited, measure.vars = c("Affected", "Unaffected"), variable.name = "Pheno", value.name = "AC")

asd[, type:="de novo"]
asd[, AC:=1]
asd[, s:=sigmoid(model_selection)]

combined = rbind(inherited, asd[, .(s, type, Pheno, AC)])
# combined[, type := factor(type, levels = c("de novo",
#                                            "inherited (AF = 0)",
#                                           "inherited (0 < AF < 1e-5)",
#                                           "inherited (1e-5 < AF < 1e-4)",
#                                           "inherited (AF > 1e-4)"))]
combined[, type:=factor(type, levels = c("de novo", "inherited"))]
ggplot(combined, aes(x = s, color = type)) + 
  facet_grid(~Pheno) + 
  geom_density() +
  theme_nature() +
#  scale_colour_viridis_d() +
  scale_x_continuous(name = "MisFit S", trans = "log10", labels = trans_format("log10", math_format(10^.x)))

save_nature("figure/inherited_DNV_distr.pdf", hw_ratio = 1/3)

inherited_n0 = 2992
inherited_n1 = 6507
asd_n0 = 5750
asd_n1 = 16876

bin_upper = 10^(seq(-3.5, -0.5, 0.5) + 0.25)
bin_upper[length(bin_upper)] = 1
bin_lower = 10^(seq(-3.5, -0.5, 0.5) - 0.25)
bin_mid = 10^(seq(-3.5, -0.5, 0.5))

summary = data.table()
for (i in 1:length(bin_mid)) {
  both_sub = both[(s<bin_upper[i]) & (s>=bin_lower[i])]
  both_m0 = both_sub[, sum(AC_Unaffected)]
  both_m1 = both_sub[, sum(AC_Affected)]
  asd_m0 = nrow(asd[(s<bin_upper[i]) & (s>=bin_lower[i]) & (Pheno == "Unaffected")])
  asd_m1 = nrow(asd[(s<bin_upper[i]) & (s>=bin_lower[i]) & (Pheno == "Affected")])

  summary = rbind(summary, data.table(
    Pheno = c("Unaffected", "Autism",  "Unaffected", "Autism"),
    type = c("de novo", "de novo", "all", "all"),
    count = c(asd_m0, asd_m1,  both_m0, both_m1),
    n = c(asd_n0, asd_n1,inherited_n0, inherited_n1),
    s_bin = bin_mid[i]
  ))
}

summary_wide1 = dcast(summary, Pheno+s_bin~type, value.var = "count")
summary_wide2 = dcast(summary, Pheno+s_bin~type, value.var = "n")
summary_wide = merge(summary_wide1, summary_wide2, by = c("Pheno", "s_bin"), suffixes = c(".count", ".n"))
summary_wide[, `de novo count per id`:= `de novo.count` / `de novo.n`]
summary_wide[, `all count per id`:= `all.count` / `all.n`]
summary_wide[, `inherited count per id`:= `all count per id` - `de novo count per id`]
summary_wide[, ratio:=`de novo count per id`/`all count per id`]

for (i in 1:nrow(summary_wide)) {
  if (!is.na(summary_wide[i, all.count])) {
  summary_wide[i, lower := rateratio.test(
    c(summary_wide[i, `de novo.count`], summary_wide[i, `all.count`]), 
    c(summary_wide[i,`de novo.n`], summary_wide[i,`all.n`]))$conf.int[1]]
    summary_wide[i, upper := rateratio.test(
      c(summary_wide[i, `de novo.count`], summary_wide[i, `all.count`]), 
      c(summary_wide[i,`de novo.n`], summary_wide[i,`all.n`]))$conf.int[2]]
  }
}

p1 = ggplot(summary_wide[is.element(Pheno, c("Unaffected", "Autism"))]) + 
  geom_point(aes(x = s_bin, y = `de novo count per id`, color = Pheno)) +
  geom_point(aes(x = s_bin, y = `inherited count per id`, color = Pheno)) +
  geom_line(aes(x = s_bin, y = `de novo count per id`, color = Pheno, linetype = "de novo")) +
  geom_line(aes(x = s_bin, y = `inherited count per id`, color = Pheno, linetype = "inherited")) +
  theme_nature() +
  scale_x_continuous(name = "MisFit S", trans = "log10", labels = trans_format("log10", math_format(10^.x))) +
  scale_y_continuous(name = "count per individual", trans = "log10", labels = trans_format("log10", math_format(10^.x))) +
  scale_linetype_manual(name = "variant", breaks = c("de novo", "inherited"), values = c("de novo" = "solid", "inherited" = "dashed")) +
  labs(color = "phenotype") +
  theme(aspect.ratio = 1)


p2 = ggplot(summary_wide[is.element(Pheno, c("Unaffected", "Autism")) & (ratio > 0)], aes(x = s_bin, y = ratio, color = Pheno)) +
  geom_abline(slope = 1, intercept = 0, color = "grey") +
  geom_point() +
  geom_line() +
  geom_errorbar(aes(ymin=lower, ymax=upper), width = 0.2) +
  theme_nature() +
  scale_x_continuous(name = "MisFit S", trans = "log10", breaks = c(0.001, 0.01, 0.1), limits = c(1e-4, 0.5), labels = trans_format("log10", math_format(10^.x))) +
  scale_y_continuous(name = "de novo ratio", trans = "log10", breaks = c(0.001, 0.01, 0.1), limits = c(1e-4, 0.5), labels = trans_format("log10", math_format(10^.x))) +
  labs(color = "") +
  theme(aspect.ratio = 1)

arrange_nature(p1, p2, widths = c(1, 1), common.legend = T, legend = "right")
save_nature("figure/count_SPARK0.pdf", hw = 1/2.5)

ASD_gene_dt = fread("SPARK_genes.txt", header = F)
ASD_gene = ASD_gene_dt$V1
NDD_gene_dt = fread("KnownGenes_20.9.24.txt")
NDD_gene = NDD_gene_dt[str_detect(Disease, "NDD"), Genes]

summary = data.table()
for (i in 1:length(bin_mid)) {
  both_sub = both[(s<bin_upper[i]) & (s>=bin_lower[i]) & is.element(Symbol, ASD_gene)]
  both_m0 = both_sub[, sum(AC_Unaffected)]
  both_m1 = both_sub[, sum(AC_Affected)]
  asd_m0 = nrow(asd[(s<bin_upper[i]) & (s>=bin_lower[i]) & (Pheno == "Unaffected") & is.element(Symbol, ASD_gene)])
  asd_m1 = nrow(asd[(s<bin_upper[i]) & (s>=bin_lower[i]) & (Pheno == "Affected") & is.element(Symbol, ASD_gene)])
  
  summary = rbind(summary, data.table(
    Pheno = c("Unaffected", "Autism",  "Unaffected", "Autism"),
    type = c("de novo", "de novo", "all", "all"),
    count = c(asd_m0, asd_m1,  both_m0, both_m1),
    n = c(asd_n0, asd_n1,inherited_n0, inherited_n1),
    s_bin = bin_mid[i]
  ))
}

summary_wide1 = dcast(summary, Pheno+s_bin~type, value.var = "count")
summary_wide2 = dcast(summary, Pheno+s_bin~type, value.var = "n")
summary_wide = merge(summary_wide1, summary_wide2, by = c("Pheno", "s_bin"), suffixes = c(".count", ".n"))
summary_wide[, `de novo count per id`:= `de novo.count` / `de novo.n`]
summary_wide[, `all count per id`:= `all.count` / `all.n`]
summary_wide[, `inherited count per id`:= `all count per id` - `de novo count per id`]
summary_wide[, ratio:=`de novo count per id`/`all count per id`]

for (i in 1:nrow(summary_wide)) {
  if (!is.na(summary_wide[i, all.count])) {
    summary_wide[i, lower := rateratio.test(
      c(summary_wide[i, `de novo.count`], summary_wide[i, `all.count`]), 
      c(summary_wide[i,`de novo.n`], summary_wide[i,`all.n`]))$conf.int[1]]
    summary_wide[i, upper := rateratio.test(
      c(summary_wide[i, `de novo.count`], summary_wide[i, `all.count`]), 
      c(summary_wide[i,`de novo.n`], summary_wide[i,`all.n`]))$conf.int[2]]
  }
}

p3 = ggplot(summary_wide[is.element(Pheno, c("Unaffected", "Autism"))]) + 
  geom_point(aes(x = s_bin, y = `de novo count per id`, color = Pheno)) +
  geom_point(aes(x = s_bin, y = `inherited count per id`, color = Pheno)) +
  geom_line(aes(x = s_bin, y = `de novo count per id`, color = Pheno, linetype = "de novo")) +
  geom_line(aes(x = s_bin, y = `inherited count per id`, color = Pheno, linetype = "inherited")) +
  theme_nature() +
  scale_x_continuous(name = "MisFit S", trans = "log10", labels = trans_format("log10", math_format(10^.x))) +
  scale_y_continuous(name = "count per individual", limits = c(1e-4, 10^3.5), trans = "log10", labels = trans_format("log10", math_format(10^.x))) +
  scale_linetype_manual(name = "variant", breaks = c("de novo", "inherited"), values = c("de novo" = "solid", "inherited" = "dashed")) +
  labs(color = "phenotype", title = "autism genes") +
  theme(aspect.ratio = 1)

summary = data.table()
for (i in 1:length(bin_mid)) {
  both_sub = both[(s<bin_upper[i]) & (s>=bin_lower[i]) & !is.element(Symbol, ASD_gene)]
  both_m0 = both_sub[, sum(AC_Unaffected)]
  both_m1 = both_sub[, sum(AC_Affected)]
  asd_m0 = nrow(asd[(s<bin_upper[i]) & (s>=bin_lower[i]) & (Pheno == "Unaffected") & !is.element(Symbol, ASD_gene)])
  asd_m1 = nrow(asd[(s<bin_upper[i]) & (s>=bin_lower[i]) & (Pheno == "Affected") & !is.element(Symbol, ASD_gene)])
  
  summary = rbind(summary, data.table(
    Pheno = c("Unaffected", "Autism",  "Unaffected", "Autism"),
    type = c("de novo", "de novo", "all", "all"),
    count = c(asd_m0, asd_m1,  both_m0, both_m1),
    n = c(asd_n0, asd_n1,inherited_n0, inherited_n1),
    s_bin = bin_mid[i]
  ))
}

summary_wide1 = dcast(summary, Pheno+s_bin~type, value.var = "count")
summary_wide2 = dcast(summary, Pheno+s_bin~type, value.var = "n")
summary_wide = merge(summary_wide1, summary_wide2, by = c("Pheno", "s_bin"), suffixes = c(".count", ".n"))
summary_wide[, `de novo count per id`:= `de novo.count` / `de novo.n`]
summary_wide[, `all count per id`:= `all.count` / `all.n`]
summary_wide[, `inherited count per id`:= `all count per id` - `de novo count per id`]
summary_wide[, ratio:=`de novo count per id`/`all count per id`]

for (i in 1:nrow(summary_wide)) {
  if (!is.na(summary_wide[i, all.count])) {
    summary_wide[i, lower := rateratio.test(
      c(summary_wide[i, `de novo.count`], summary_wide[i, `all.count`]), 
      c(summary_wide[i,`de novo.n`], summary_wide[i,`all.n`]))$conf.int[1]]
    summary_wide[i, upper := rateratio.test(
      c(summary_wide[i, `de novo.count`], summary_wide[i, `all.count`]), 
      c(summary_wide[i,`de novo.n`], summary_wide[i,`all.n`]))$conf.int[2]]
  }
}

p4 = ggplot(summary_wide[is.element(Pheno, c("Unaffected", "Autism"))]) + 
  geom_point(aes(x = s_bin, y = `de novo count per id`, color = Pheno)) +
  geom_point(aes(x = s_bin, y = `inherited count per id`, color = Pheno)) +
  geom_line(aes(x = s_bin, y = `de novo count per id`, color = Pheno, linetype = "de novo")) +
  geom_line(aes(x = s_bin, y = `inherited count per id`, color = Pheno, linetype = "inherited")) +
  theme_nature() +
  scale_x_continuous(name = "MisFit S", trans = "log10", labels = trans_format("log10", math_format(10^.x))) +
  scale_y_continuous(name = "count per individual", limits = c(1e-4, 10^3.5), trans = "log10", labels = trans_format("log10", math_format(10^.x))) +
  scale_linetype_manual(name = "variant", breaks = c("de novo", "inherited"), values = c("de novo" = "solid", "inherited" = "dashed")) +
  labs(color = "phenotype", title = "other genes") +
  theme(aspect.ratio = 1)


arrange_nature(p3, p4, common.legend = T, legend = "right")
save_nature("figure/count_SPARK0_gene.pdf", hw_ratio = 1/2.5)

