library(data.table)
library(ggplot2)
library(actuar)
library(viridis)
source("../../settings/plot_settings.R")

dt = fread("PIG_stats.csv")

all_mu = c(1e-8, 1e-7)
all_s = c(1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.9)
all_nsample = c(25, 100, 400, 1600)
all_ngenome = c(40000, 200000, 1000000)
ngroup = 400

is_best = function(s0, AC_sample) {
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

rate_best = function(s0, AF, nsample) {
  count = c(0, 0, 0)
  for (i in 1:ngroup) {
    AF_sample = sample(AF, nsample)
    AC_sample = rbinom(nsample, 2 * ngenome, AF_sample)
    AC_sample = pmin(AC_sample, ngenome)
    est = is_best(s0, AC_sample)
    count[est] = count[est] + 1
  }
  return(count/ngroup)
}

summary = data.table()

for (mu0 in all_mu) {
  data = fread(paste0("sim_q/simulation_", mu0, ".csv"))
  colnames(data) = c("mu", "s", "AF", "Occ")
  for (s0 in all_s[2:(length(all_s)-1)]) {
    AF = data[s==s0, rep(AF, Occ)]
    AF = AF[AF<=0.5]
    for (ngenome in all_ngenome) {
      for (nsample in all_nsample) {
        rate = rate_best(s0, AF, nsample)
        summary = rbind(summary, data.table(mu = mu0, s = s0, ngenome = ngenome, nsample = nsample, 
                                            accuracy = rate[2], underestimation = rate[1], overestimation = rate[3]))
      }
    }
  }
}

fwrite(summary, "est_summary.csv")

summary[, s:=format(s, scientific = FALSE, drop0trailing = T)]

p1 = ggplot(summary, aes(x = factor(nsample), y = accuracy, color = factor(ngenome, labels = c('40K', '200K', '1M')))) + 
  theme_nature() + 
  geom_line(aes(group = ngenome + mu, linetype = factor(mu))) + 
  facet_grid(~s) +
  scale_color_manual(name  = "# genomes", values = mako(3, alpha = 1, begin = 0.2, end = 0.8, direction = -1)) +
  scale_x_discrete(name = "# variants per group") +
  scale_linetype(name = "mutation rate") +
  ggtitle("simulated s") +
  theme(legend.position="bottom", legend.box = "vertical", 
        legend.box.spacing = unit(0, "pt"),
        legend.spacing.y = unit(-6, 'pt'),
        plot.tag = element_text(margin = margin(0,0,-11,0))
        )

save_nature(p1 + labs(tag = "c"), filename = "figure/est_agg_accurate.eps", hw_ratio = 1/2, width_ratio = 1)

p2 = ggplot(summary, aes(x = factor(nsample), y = overestimation, color = factor(ngenome, labels = c('40K', '200K', '1M')))) + 
  theme_nature() + 
  geom_line(aes(group = ngenome + mu, linetype = factor(mu))) + 
  facet_grid(~s) +
  scale_color_manual(name  = "# genomes", values = mako(3, alpha = 1, begin = 0.2, end = 0.8, direction = -1)) +
  scale_x_discrete(name = "# variants per group") +
  scale_y_continuous(limits = c(0, 1)) +
  scale_linetype(name = "mutation rate") +
  theme(legend.position="bottom", legend.box = "vertical", 
        legend.box.spacing = unit(0, "pt"),
        legend.spacing.y = unit(-6, 'pt')) +
  ggtitle("simulated s")

p3 = ggplot(summary, aes(x = factor(nsample), y = underestimation, color = factor(ngenome, labels = c('40K', '200K', '1M')))) + 
  theme_nature() + 
  geom_line(aes(group = ngenome + mu, linetype = factor(mu))) + 
  facet_grid(~s) +
  scale_color_manual(name  = "# genomes", values = mako(3, alpha = 1, begin = 0.2, end = 0.8, direction = -1)) +
  scale_x_discrete(name = "# variants per group") +
  scale_y_continuous(limits = c(0, 1)) +
  scale_linetype(name = "mutation rate") +
  theme(legend.position="bottom", legend.box = "vertical", 
        legend.box.spacing = unit(0, "pt"),
        legend.spacing.y = unit(-6, 'pt')) +
  ggtitle("simulated s")

arrange_nature(p1, p2, p3, 
               common.legend = T, legend = "bottom", nrow = 3, ncol = 1)
save_nature(filename = "figure/est_agg.eps", hw_ratio = 2.5/2, width_ratio = 1)


