library(data.table)
library(ggplot2)
library(actuar)
library(viridis)
library(ggpubr)
library(scales)
source("../../settings/plot_settings.R")
dnbinom2 = function(AC, an, alpha, beta, log = F) {
  dnbinom(AC, size = alpha, prob = 1 - 1/((beta/an) + 1), log = log)
}

pnbinom2 = function(AC, an, alpha, beta) {
  pnbinom(AC, size = alpha, prob = 1 - 1/((beta/an) + 1))
}

dt = fread("PIG_stats.csv")
Ns = 400000

# PIG
# model = "PIG"
# dt[, cum_0 := ppoisinvgauss(0 * Ns, fit_ig_mu * Ns, fit_ig_lambda * Ns)]
# dt[, cum_1 := ppoisinvgauss(1e-5 * Ns, fit_ig_mu * Ns, fit_ig_lambda * Ns)]
# dt[, cum_2:= ppoisinvgauss(1e-4 * Ns, fit_ig_mu * Ns, fit_ig_lambda * Ns)]
# dt[, cum_3:= ppoisinvgauss(1e-3 * Ns, fit_ig_mu * Ns, fit_ig_lambda * Ns)]
Ne = 1e6
model = paste0("NB_", Ne)
dt[, cum_0 := pnbinom2(0, Ns, 4*Ne*mu, 4*Ne*s)]
dt[, cum_1 := pnbinom2(1e-5 * Ns, Ns, 4*Ne*mu, 4*Ne*s)]
dt[, cum_2 := pnbinom2(1e-4 * Ns, Ns, 4*Ne*mu, 4*Ne*s)]
dt[, cum_3 := pnbinom2(1e-3 * Ns, Ns, 4*Ne*mu, 4*Ne*s)]

p1 = ggplot(dt, aes(x = s, y = pAF0, color = factor(mu))) + geom_point() + 
  scale_color_brewer(name = "mutation rate", palette="YlOrRd") +
  scale_y_continuous(name = "0", limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  theme_nature() + 
  geom_line(aes(x = s, y = cum_0)) +
  scale_x_continuous(name = bquote(s), trans = "log10", labels = trans_format("log10", math_format(10^.x))) +
  theme(legend.position = "none")

p2 = ggplot(dt, aes(x = s, y = pAF1, color = factor(mu))) + geom_point() + 
  scale_color_brewer(name = "mutation rate", palette="YlOrRd") +
  scale_y_continuous(name = "(0, 1e-5]", limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  theme_nature() + 
  geom_line(aes(x = s, y = cum_1 - cum_0)) +
  scale_x_continuous(name = bquote(s), trans = "log10", labels = trans_format("log10", math_format(10^.x))) +
  theme(legend.position = "none")

p3 = ggplot(dt, aes(x = s, y = pAF2, color = factor(mu))) + geom_point() + 
  scale_color_brewer(name = "mutation rate", palette="YlOrRd") +
  scale_y_continuous(name = "(1e-5, 1e-4]", limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  theme_nature() + 
  geom_line(aes(x = s, y = cum_2 - cum_1)) +
  scale_x_continuous(name = bquote(s), trans = "log10", labels = trans_format("log10", math_format(10^.x))) +
  theme(legend.position = "none")

p4 = ggplot(dt, aes(x = s, y = pAF3, color = factor(mu))) + geom_point() + 
  scale_color_brewer(name = "mutation rate", palette="YlOrRd") +
  scale_y_continuous(name = "(1e-4, 1e-3]", limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  theme_nature() + 
  geom_line(aes(x = s, y = cum_3 - cum_2)) +
  scale_x_continuous(name = bquote(s), trans = "log10", labels = trans_format("log10", math_format(10^.x))) +
  theme(legend.position = "none")

p5 = ggplot(dt, aes(x = s, y = pAF4 + pAF5, color = factor(mu))) + geom_point() + 
  scale_color_brewer(name = "mutation rate", palette="YlOrRd") +
  scale_y_continuous(name = "(1e-3, 1]", limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  theme_nature() + 
  geom_line(aes(x = s, y = 1 - cum_3)) +
  scale_x_continuous(name = bquote(s), trans = "log10", labels = trans_format("log10", math_format(10^.x)))

legend = get_legend(p5)
p5 = p5 + theme(legend.position = "none")

p = arrange_nature(p1,p2,p3,p4,p5,legend, ncol = 3, nrow = 2, labels = NULL)
annotate_figure(p, left = text_grob("P (sample allele frequency in range)", rot = 90, size = 7 * custom_expand_ratio))
save_nature(paste0("figure/pAF_", model, ".eps"), hw_ratio = 2/3, width_ratio = 1, width = "single")


mu0 = 1e-8
dt2 = dt[s>0&mu==mu0]
dt2[, simulate:=mean]
dt2[, Gamma_1:=mu0/s]
dt2[, Gamma_2:=mu0/s]
dt2[, Gamma_3:=mu0/s]
dt2[, IG:=fit_ig_mu]
dt2[, const:=mu0/s]
dt2 = dt2[, .(s, simulate, Gamma_1, Gamma_2, Gamma_3, IG, const)]
dt2 = melt(dt2, id.vars = "s", measure.vars = c("simulate", "Gamma_1", "Gamma_2", "Gamma_3", "IG", "const"), 
           variable.name = "model", value.name = "AFmean")

dt2[, model := factor(model, 
                      levels = c("simulate", "Gamma_1", "Gamma_2", "Gamma_3", "IG", "const"),
                      labels = c("Simulated", "NB (Ne=1e4)", "NB (Ne=1e5)", "NB (Ne=1e6)", "PIG", "Poisson")
)]

p1 = ggplot(dt2) + 
  geom_line(aes(x = s, y = AFmean, color = model), position=position_jitter(w=0.05, h=0.05)) +
  theme_nature() + 
  scale_color_manual(name = "", values = c("black", "yellow", "orange", "red", "blue", "brown")) +
  scale_x_continuous(name = bquote(s), trans = "log10", labels = trans_format("log10", math_format(10^.x))) +
  scale_y_continuous(name = "population AF mean", trans = "log10", labels = trans_format("log10", math_format(10^.x)))

mu0 = 1e-8
dt2 = dt[s>0&mu==mu0]
dt2[, simulate:=var]
dt2[, Gamma_1:=mu0/s^2/4/10000]
dt2[, Gamma_2:=mu0/s^2/4/100000]
dt2[, Gamma_3:=mu0/s^2/4/1000000]
dt2[, IG:=fit_ig_mu^3/fit_ig_lambda]
dt2[, const:=0]
dt2 = dt2[, .(s, simulate, Gamma_1, Gamma_2, Gamma_3, IG, const)]
dt2 = melt(dt2, id.vars = "s", measure.vars = c("simulate", "Gamma_1", "Gamma_2", "Gamma_3", "IG", "const"), 
           variable.name = "model", value.name = "AFvar")

dt2[, model := factor(model, 
                      levels = c("simulate", "Gamma_1", "Gamma_2", "Gamma_3", "IG", "const"),
                      labels = c("Simulated", "NB (Ne=1e4)", "NB (Ne=1e5)", "NB (Ne=1e6)", "PIG", "Poisson")
                      )]
dt2[, AFvar:=pmax(AFvar, 1e-18)]
p2 = ggplot(dt2) + 
  geom_line(aes(x = s, y = AFvar, color = model)) +
  theme_nature() + 
  scale_color_manual(name = "", values = c("black", "yellow", "orange", "red", "blue", "brown")) +
  scale_x_continuous(name = bquote(s), trans = "log10", labels = trans_format("log10", math_format(10^.x))) +
  scale_y_continuous(name = "population AF variance", trans = "log10", labels = trans_format("log10", math_format(10^.x)))
arrange_nature(p1, p2, ncol = 2, common.legend = T, legend = "right")
save_nature(paste0("figure/popvar_", mu0, ".eps"), hw_ratio = 1/2.5, width_ratio = 1)


mu0 = 1e-8
dt2 = dt[s>0&mu==mu0]
dt2[, simulate:=meansample]
dt2[, Gamma_1:=mu0/s]
dt2[, Gamma_2:=mu0/s]
dt2[, Gamma_3:=mu0/s]
dt2[, IG:=fit_ig_mu]
dt2[, const:=mu0/s]
dt2 = dt2[, .(s, simulate, Gamma_1, Gamma_2, Gamma_3, IG, const)]
dt2 = melt(dt2, id.vars = "s", measure.vars = c("simulate", "Gamma_1", "Gamma_2", "Gamma_3", "IG", "const"), 
           variable.name = "model", value.name = "AFmean")

dt2[, model := factor(model, 
                      levels = c("simulate", "Gamma_1", "Gamma_2", "Gamma_3", "IG", "const"),
                      labels = c("Simulated", "NB (Ne=1e4)", "NB (Ne=1e5)", "NB (Ne=1e6)", "PIG", "Poisson")
)]

p3 = ggplot(dt2) + 
  geom_line(aes(x = s, y = AFmean, color = model), position=position_jitter(w=0.05, h=0.05)) +
  theme_nature() + 
  scale_color_manual(name = "", values = c("black", "yellow", "orange", "red", "blue", "brown")) +
  scale_x_continuous(name = bquote(s), trans = "log10", labels = trans_format("log10", math_format(10^.x))) +
  scale_y_continuous(name = "sample AF mean", trans = "log10", labels = trans_format("log10", math_format(10^.x)))

dt2 = dt[s>0&mu==mu0]
dt2[, simulate:=varsample]
Ne = 10000
dt2[, Gamma_1:=4*Ne*mu*1/((4*Ne*s/Ns) + 1)/(1 - 1/((4*Ne*s/Ns) + 1))^2 / (Ns)^2]
Ne = 100000
dt2[, Gamma_2:=4*Ne*mu*1/((4*Ne*s/Ns) + 1)/(1 - 1/((4*Ne*s/Ns) + 1))^2 / (Ns)^2]
Ne = 1000000
dt2[, Gamma_3:=4*Ne*mu*1/((4*Ne*s/Ns) + 1)/(1 - 1/((4*Ne*s/Ns) + 1))^2 / (Ns)^2]
dt2[, const:=mu/s*Ns / (Ns)^2]

dt2[, IG:=fit_ig_mu^3/fit_ig_lambda + fit_ig_mu/Ns]

dt2 = dt2[, .(s, simulate, Gamma_1, Gamma_2, Gamma_3, IG, const)]
dt2 = melt(dt2, id.vars = "s", measure.vars = c("simulate", "Gamma_1", "Gamma_2", "Gamma_3", "IG",  "const"), 
           variable.name = "model", value.name = "AFvar")

dt2[, model := factor(model, 
                      levels = c("simulate", "Gamma_1", "Gamma_2", "Gamma_3", "IG", "const"),
                      labels = c("Simulated", "NB (Ne=1e4)", "NB (Ne=1e5)", "NB (Ne=1e6)", "PIG", "Poisson")
)]
dt2[, AFvar:=pmax(AFvar, 1e-18)]
p4 = ggplot(dt2) + 
  geom_line(aes(x = s, y = AFvar, color = model)) +
  theme_nature() + 
  scale_color_manual(name = "", values = c("black", "yellow", "orange", "red", "blue", "brown")) +
  scale_x_continuous(name = bquote(s), trans = "log10", labels = trans_format("log10", math_format(10^.x))) +
  scale_y_continuous(name = "sample AF variance", trans = "log10", labels = trans_format("log10", math_format(10^.x)))
arrange_nature(p3, p4, ncol = 2, common.legend = T, legend = "right")
save_nature("figure/samplevar_1e-8.eps", hw_ratio = 1/2.5, width_ratio = 1)

