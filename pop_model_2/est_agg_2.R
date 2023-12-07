library(data.table)
library(ggplot2)
library(actuar)
library(viridis)
library(scales)
source("../../settings/plot_settings.R")

dt = fread("PIG_stats.csv")

all_mu = c(1e-8, 1e-7)
all_s = c(1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.9)
all_nsample = c(25, 100, 400, 1600)
all_ngenome = c(40000, 200000, 1000000)
ngroup = 400

is_best = function(s0, AC_sample, ngenome, mu0) {
  max_likelihood = -Inf
  best_s = NA
  for (s1 in all_s) {
    if ((s1/s0 > 10.01) | (s1/s0 < 0.099)) {
      next
    }
    fit_ig_mu = dt[(mu==mu0)&(s==s1), fit_ig_mu]
    fit_ig_lambda = dt[(mu==mu0)&(s==s1), fit_ig_lambda]
    likelihood = mean(dpoisinvgauss(AC_sample, fit_ig_mu * 2 * ngenome, fit_ig_lambda * 2 * ngenome, log = T))
    if (likelihood > max_likelihood) {
      best_s = s1
      max_likelihood = likelihood
    }
  }
  if (best_s < s0)
    return(1)
  if (best_s == s0)
    return(2)
  if (best_s > s0)
    return(3)

}

rate_best = function(s0, AF1, AF2, nsample, ngenome, mu0) {
  count = c(0, 0, 0)
  for (i in 1:ngroup) {
    sample_id = sample(1:length(AF1), nsample)
    AF1_sample = AF1[sample_id]
    AC1_sample = rbinom(nsample, 2 * ngenome, AF1_sample)
    AC1_sample = pmin(AC1_sample, ngenome)
    if (!is.null(AF2)) {
      AF2_sample = AF2[sample_id]
      AC2_sample = rbinom(nsample, 2 * ngenome, AF2_sample)
      AC2_sample = pmin(AC2_sample, ngenome)
      AC_sample = AC1_sample + AC2_sample
      est = is_best(s0, AC_sample, 2 * ngenome, mu0)
    }
    else {
      AC_sample = AC1_sample
      est = is_best(s0, AC_sample, ngenome, mu0)
    }
    count[est] = count[est] + 1
  }
  return(count/ngroup)
}

summary = data.table()

for (mu0 in all_mu) {
  data = fread(paste0("sim_q_two_pop/simulation_", mu0, ".csv"))
  colnames(data) = c("mu", "s", "AF1", "AF2", "AF3", "AF4")
  data = data[AF1<0.5&AF2<0.5]
  for (s0 in all_s[2:(length(all_s)-1)]) {
    AF = data[s==s0, .(AF1, AF2)]
    for (nsample in all_nsample) {
      for (ngenome in all_ngenome)
        {
      rate = rate_best(s0, AF[, AF1], NULL, nsample, ngenome, mu0)
      summary = rbind(summary, data.table(mu = mu0, s = s0, type = 1, ngenome = ngenome, nsample = nsample, 
                                          accuracy = rate[2], underestimation = rate[1], overestimation = rate[3]))
      rate = rate_best(s0, AF[, AF1], AF[, AF2], nsample, ngenome / 2, mu0)
      summary = rbind(summary, data.table(mu = mu0, s = s0, type = 2, ngenome = ngenome, nsample = nsample, 
                                          accuracy = rate[2], underestimation = rate[1], overestimation = rate[3]))
      }
    }
  }
}

fwrite(summary, "est_summary_2.csv")
summary = fread("est_summary_2.csv")

summary[, s:=format(s, scientific = FALSE, drop0trailing = T)]
summary[, ngenome:=factor(ngenome, levels = all_ngenome, labels = c('40K', '200K', '1M'))]
summary[, populations:=factor(type, levels = c(1, 2))]


p1 = ggplot(summary, aes(x = ngenome, y = accuracy, color = factor(nsample), linetype = populations)) + 
  theme_nature() + 
  geom_line(aes(group = interaction(nsample, populations))) + 
  facet_grid(mu~s, labeller = label_bquote(cols = "s"==.(s), rows = nu==.(mu))) +
  scale_color_manual(name  = "# variants per group", values = magma(4, alpha = 1, begin = 0.1, end = 0.9, direction = -1)) +
  scale_x_discrete(name = "# genomes") +
  scale_y_continuous(limits = c(0, 1), minor_breaks = NULL, breaks = seq(0, 1, 0.2)) +
  scale_linetype(name = "# populations") +
  theme(legend.position="bottom", legend.box = "vertical", 
        legend.box.spacing = unit(0, "pt"),
        legend.spacing.y = unit(-6, 'pt'),
        plot.tag = element_text(margin = margin(0,0,-11,0))
  )

p2 = ggplot(summary, aes(x = ngenome, y = overestimation, color = factor(nsample), linetype = populations)) + 
  theme_nature() + 
  geom_line(aes(group = interaction(nsample, populations))) + 
  facet_grid(mu~s, labeller = label_bquote(cols = "s"==.(s), rows = nu==.(mu))) +
  scale_color_manual(name  = "# variants per group", values = magma(4, alpha = 1, begin = 0.1, end = 0.9, direction = -1)) +
  scale_x_discrete(name = "# genomes") +
  scale_y_continuous(limits = c(0, 1), minor_breaks = NULL, breaks = seq(0, 1, 0.2)) +
  scale_linetype(name = "# populations") +
  theme(legend.position="bottom", legend.box = "vertical", 
        legend.box.spacing = unit(0, "pt"),
        legend.spacing.y = unit(-6, 'pt'),
        plot.tag = element_text(margin = margin(0,0,-11,0))
  )

p3 = ggplot(summary, aes(x = ngenome, y = underestimation, color = factor(nsample), linetype = populations)) + 
  theme_nature() + 
  geom_line(aes(group = interaction(nsample, populations))) + 
  facet_grid(mu~s, labeller = label_bquote(cols = "s"==.(s), rows = nu==.(mu))) +
  scale_color_manual(name  = "# variants per group", values = magma(4, alpha = 1, begin = 0.1, end = 0.9, direction = -1)) +
  scale_x_discrete(name = "# genomes") +
  scale_y_continuous(limits = c(0, 1), minor_breaks = NULL, breaks = seq(0, 1, 0.2)) +
  scale_linetype(name = "# populations") +
  theme(legend.position="bottom", legend.box = "vertical", 
        legend.box.spacing = unit(0, "pt"),
        legend.spacing.y = unit(-6, 'pt'),
        plot.tag = element_text(margin = margin(0,0,-11,0))
  )


arrange_nature(p2, p3, 
               common.legend = T, legend = "bottom", nrow = 1, ncol = 2)
save_nature(filename = "figure/est_agg_bipop.pdf", hw_ratio = 1/3, width_ratio = 2)
save_nature(p1 + labs(tag = "c"), filename = "figure/est_agg_bipop_accurate.pdf", hw_ratio = 1/1.5, width_ratio = 1)


legend = get_legend(p1 +   theme(legend.position="right", legend.box = "vertical", 
                                 legend.box.spacing = unit(0, "pt"),
                                 legend.spacing.y = unit(6, 'pt'),
                                 plot.tag = element_text(margin = margin(0,0,0,0))
))
arrange_nature(p1 + theme(legend.position = "none"), p2 + theme(legend.position = "none"), 
               p3 + theme(legend.position = "none"), legend, nrow = 2, ncol = 2, labels = c("a", "b", "c"))
save_nature(filename = "figure/est_agg_bipop_all.pdf", hw_ratio = 2/3, width_ratio = 2)


