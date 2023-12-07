library(data.table)
library(ggplot2)
source("../../settings/plot_settings.R")

dt = fread("syn_summary.txt")
sample_dt = dt[ne=="sample"]
dt = dt[ne!="sample"]
dt[, ne:=as.numeric(ne)]
p1 = ggplot(dt[upper == 0], aes(y = prob_minor, x = ne)) +
  theme_nature() +
  geom_point() + geom_path() +
  geom_hline(data = sample_dt[upper == 0], aes(yintercept = prob_minor), color = "red", linetype = "dashed") +
  labs(x = "final Ne", y = "probability", title = "random")

dt = fread("syn_summary_highmu.txt")
sample_dt = dt[ne=="sample"]
dt = dt[ne!="sample"]
dt[, ne:=as.numeric(ne)]
p2 = ggplot(dt[upper == 0], aes(y = prob_minor, x = ne)) +
  theme_nature() +
  geom_point() + geom_path() +
  geom_hline(data = sample_dt[upper == 0], aes(yintercept = prob_minor), color = "red", linetype = "dashed") +
  labs(x = "final Ne", y = "probability", title = "high mutation rate")

arrange_nature(p1, p2)
save_nature("figure/final_ne.pdf", hw_ratio = 1/2)
