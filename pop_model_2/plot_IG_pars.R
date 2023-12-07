library(data.table)
library(ggplot2)
library(scales)
source("../../settings/plot_settings.R")

dt = fread("PIG_stats.csv")

p1 = ggplot(dt, aes(x = s, y = mean, color = factor(mu))) + 
  geom_line(aes(x = s, y = fit_ig_mu), linetype = "dashed") +
  geom_point() + 
  theme_nature() +
  scale_color_brewer(name = "mutation rate", palette="YlOrRd") +
  scale_x_continuous(name = bquote(s), trans = "log10", labels = trans_format("log10", math_format(10^.x))) +
  scale_y_continuous(name = bquote(mu[IG]), trans = "log10", labels = trans_format("log10", math_format(10^.x)))

p2 = ggplot(dt, aes(x = s, y = ig_lambda, color = factor(mu))) + 
  geom_line(aes(x = s, y = fit_ig_lambda), linetype = "dashed") +
  geom_point() + 
  theme_nature() +
  scale_color_brewer(name = "mutation rate", palette="YlOrRd") +
  scale_x_continuous(name = bquote(s), trans = "log10", labels = trans_format("log10", math_format(10^.x))) +
  scale_y_continuous(name = bquote(lambda[IG]), trans = "log10", labels = trans_format("log10", math_format(10^.x)))

options(scipen = 0)

arrange_nature(p1, p2, ncol = 2, common.legend = T, legend = "right")
save_nature("figure/IG-pars-to-s.pdf", hw_ratio = 1/2.5, width_ratio = 1)


