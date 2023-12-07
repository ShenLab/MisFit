library(data.table)
library(ggplot2)
library(actuar)
library(viridis)
library(scales)
source("../../settings/plot_settings.R")

Ns = 4e5

logit = function(x) {
  return (log(x/(1-x)))
}

expit = function(x) {
  return(1/(1+exp(-x)))
}

softminus = function(x) {
  return(-log(1+exp(-x)))
}

softplus = function(x) {
  return(log(1+exp(x)))
}

# individual missense
dt = data.table()
for (mu0 in c(1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7)) {
  data = fread(paste0("sim_q/simulation_", mu0, ".csv"))
  colnames(data) = c("mu", "s", "AF", "Occ")

  for (s0 in c(1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 0.5, 0.7, 0.9)) {
    AF = data[s==s0, rep(AF, Occ)]
    AFadj = pmin(AF, 0.5)
    ig_mu = mean(AFadj)
    ig_lambda = length(AFadj) / sum(1 / AFadj - 1 / ig_mu)
    
    AFsample = rbinom(length(AFadj), Ns, AFadj) / Ns
    pAF0 = sum(AFsample == 0) / length(AFsample)
    pAF1 = sum((AFsample > 0) & (AFsample <= 1e-5 )) / length(AFsample)
    pAF2 = sum((AFsample > 1e-5) & (AFsample <= 1e-4)) / length(AFsample)
    pAF3 = sum((AFsample > 1e-4) & (AFsample <= 1e-3)) / length(AFsample)
    pAF4 = sum((AFsample > 1e-3) & (AFsample <= 1e-2)) / length(AFsample)
    pAF5 = sum((AFsample > 1e-2)) / length(AFsample)
    
    
    dt = rbind(dt, data.table(mu=mu0, s=s0, 
                              mean=mean(AFadj), var=var(AFadj), 
                              varsample = var(AFsample),
                              meansample = mean(AFsample),
                              ig_mu = ig_mu, ig_lambda = ig_lambda,
                              pAF0 = pAF0, pAF1 = pAF1, pAF2=pAF2, pAF3 = pAF3, 
                              pAF4 = pAF4, pAF5 = pAF5
    ))
  }
}


# fit mean
x = dt[, logit(s)]
y = dt[, log(ig_mu / mu)]

nlsfit <- nls(y ~ -log(1 + exp(x + a)) + a, start = list(a = 10))
a = nlsfit$m$getPars()["a"]
f_logmean = function(logmu, logits) {
  return(-log(1 + exp(logits + a)) + a + logmu)
}
dt[, fit_ig_mu := exp(f_logmean(log(mu), logit(s)))]

# fit lambda
x = dt[s<0.1, log(mu)]
z = dt[s<0.1, pmin(log(ig_lambda), 0)]

nlsfit <- nls(z ~ (c * x^2 + d * x + e),
              start=list(c = 0, d = 6, e = 50))

c = nlsfit$m$getPars()["c"]
d = nlsfit$m$getPars()["d"]
e = nlsfit$m$getPars()["e"]



f_loglambda = function(logmu) {
  return((c * logmu ^ 2 + d * logmu + e))
}

dt[, fit_ig_lambda := exp(f_loglambda(log(mu)))]

fwrite(dt, file = "PIG_stats.csv")

likedt = data.table()
for (Ns in c(4e5))
  for (mu0 in c(1e-8, 1e-7)) {
    for (s0 in 10^seq(-6, -0.01, 0.1)) {
      af = c(0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
      ac = round(af * Ns)
      like = dpoisinvgauss(ac, exp(f_logmean(log(mu0), logit(s0))) * Ns, exp(f_loglambda(log(mu0))) * Ns, log = T)
      likedt = rbind(likedt, data.table(af, like, s=s0, Ns=Ns, mu=mu0))
    }
  }

max_likedt = likedt[, maxlike := max(like) , by = .(af, mu, Ns)]

options(scipen = 9)

show_ylim = -30
Ns0 = 4e5

p1 = ggplot(likedt[(mu == 1e-8) & (Ns == Ns0)], aes(x = s, y = pmax(like, show_ylim), color = factor(af))) + 
  geom_line() +
  theme_nature() +
  geom_point(data = likedt[(mu == 1e-8) & (Ns == Ns0) & (like == maxlike)]) +
  scale_x_continuous(name = bquote(s), trans = "log10", labels = trans_format("log10", math_format(10^.x))) +
  scale_y_continuous(name = "log likelihood", limits = c(show_ylim, 0)) +
  scale_color_viridis(name = "sample AF", discrete = T, option = "D") +
  ggtitle("mutation rate = 1e-8")
  
p2 = ggplot(likedt[(mu == 1e-7) & (Ns == Ns0)], aes(x = s, y = pmax(like, show_ylim), color = factor(af))) + 
  geom_line() +
  theme_nature() +
  geom_point(data = likedt[(mu == 1e-7) & (Ns == Ns0) & (like == maxlike)]) +
  scale_x_continuous(name = bquote(s), trans = "log10", labels = trans_format("log10", math_format(10^.x))) +
  scale_y_continuous(name = "log likelihood", limits = c(show_ylim, 0)) +
  scale_color_viridis(name = "sample AF", discrete = T, option = "D") +
  ggtitle("mutation rate = 1e-7")

arrange_nature(p1, p2, common.legend = T, legend = "bottom")
save_nature("figure/like_200k.pdf", hw_ratio = 2/3)

likedt = data.table()
  for (mu0 in c(1e-8, 1e-7)) {
    for (s0 in c(1e-5, 1e-4, 1e-3, 1e-2, 0.1)) {
      af = 10^seq(-5, 0, 0.2)
      like = pinvgauss(af, mean = exp(f_logmean(log(mu0), logit(s0))), shape = exp(f_loglambda(log(mu0))), log = T)
      likedt = rbind(likedt, data.table(af, like, s=s0, mu=mu0))
    }
  }


p3 = ggplot(likedt[(mu == 1e-8)], aes(x = af, y = like, color = factor(s))) + 
  geom_line() +
  theme_nature() +
  scale_x_continuous(name = "allele frequency", trans = "log10", labels = trans_format("log10", math_format(10^.x))) +
  scale_y_continuous(name = "log cumulative probability") +
  scale_color_viridis(name = "s", discrete = T, option = "D") +
  ggtitle("mutation rate = 1e-8")

p4 = ggplot(likedt[(mu == 1e-7)], aes(x = af, y = like, color = factor(s))) + 
  geom_line() +
  theme_nature() +
  scale_x_continuous(name = "allele frequency", trans = "log10", labels = trans_format("log10", math_format(10^.x))) +
  scale_y_continuous(name = "log cumulative probability") +
  scale_color_viridis(name = "s", discrete = T, option = "D") +
  ggtitle("mutation rate = 1e-7")

arrange_nature(p3, p4, common.legend = T, legend = "bottom")
save_nature("figure/af_cdf.pdf", hw_ratio = 2/3)

