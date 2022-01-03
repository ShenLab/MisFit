library(data.table)
library(ggplot2)
options(scipen=1)

logit = function(x) {
  return (log(x/(1-x)))
}

fitn = function(x) {
  a = 0.001044664
  b = 0.03358582
  c = 0.4557228
  d = 16.08522
  x = pmin(x, 4)
  x = pmax(x, -13.3)
  fitn0  = function(x) {
    return(a * x ^ 3 + b * x ^ 2 + c * x + d)
  }
  
  return(sapply(x, fitn0))
}

rarethresh = 0.1
times = 100
all_nsample = c(50, 100, 200, 400, 800, 1600)
Ne = 3386437

for (an in c(400000, 2000000, 10000000))
for (mu in c(1.6e-8, 1.6e-7)) {
  data = fread(paste0("sim_q/simulation_", mu, ".csv"))
  colnames(data) = c("mu", "s", "AC", "Occ")
  data[, AF:=AC/2/Ne]
  dt_fits = data.table()
  for (s1 in c(0.00016, 0.0016, 0.016, 0.16)) {
    AF = data[s==s1, rep(AF, Occ)]
    #AF = AF[AF<rarethresh]
    k = rbinom(length(AF), an, AF)
    k = pmin(k, rarethresh * an)
    for (nsample in all_nsample) {
      for (i in 1:times){
        samples = sample(k, nsample)
        
        f = function(s) {
          ne = exp(fitn(logit(s)))
          return(-sum(dnbinom(samples, size = 4*mu*ne, prob = 1 - 1 / (4*s*ne/an + 1), log = T)))
        }
        s_est = min(mu/mean(samples)*an, 1)
        s_fit = optim(s_est, f, method = "Brent", lower = 0, upper = 1)$par
        dt_fits = rbind(dt_fits, data.table(s_sim = s1, s_fit = s_fit, s_est = s_est, nsample = nsample))
      }
    }
  }
  
  
  ggplot(dt_fits, aes(x = factor((nsample)), y = log10(s_fit))) +
    theme_minimal(base_size = 14) +
    facet_wrap(~ s_sim, nrow = 1) +
    geom_hline(aes(yintercept = log10(s_sim)), linetype = "dashed", color = "grey") +
    geom_violin(fill = "royalblue", color = "royalblue") +
    xlab("number of variants") + 
    ylab(expression(log[10]~s[MLE])) +
    ylim(-6, 0) +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
  ggsave(paste0("comp_mle_", an, "_", mu, ".svg"), width = 18, height = 6, units = "cm")
  
  
  
  
# like_diff = data.table()
# for (s1 in c(0.000016, 0.00016, 0.0016, 0.016)) {
#   s2 = signif(s1 * 10, 2)
#   alpha1 = 4 * mu * exp(fitn(logit(s1)))
#   beta1 = 4 * s1 * exp(fitn(logit(s1)))
#   alpha2 = 4 * mu * exp(fitn(logit(s2)))
#   beta2 = 4 * s2 * exp(fitn(logit(s2)))
#   
#   
#   AF = data[s==s1, rep(AF, Occ)]
#   k = rbinom(length(AF), an, AF)
#   k = pmin(k, rarethresh * an)
#   for (nsample in all_nsample) {
#     for (i in 1:times) {
#       samples = sample(k, nsample)
#       #samples = rnbinom(nsample, size = alpha1, prob = 1 - 1/( beta1/an + 1))
#       #samples = samples[samples<rarethresh*an]
#       
#       like1 = sum(dnbinom(samples, size = alpha1, prob = 1 - 1/( beta1/an + 1), log = T))
#       like2 = sum(dnbinom(samples, size = alpha2, prob = 1 - 1/( beta2/an + 1), log = T))
#       like_diff = rbind(like_diff, data.table(nsample = nsample, diff = like1 - like2, s1 = s1, s2 = s2))
#     }
#   }
#   
#   AF = data[s==s2, rep(AF, Occ)]
#   k = rbinom(length(AF), an, AF)
#   k = pmin(k, rarethresh * an)
#   for (nsample in all_nsample) {
#     for (i in 1:times) {
#       samples = sample(k, nsample)
#       #samples = rnbinom(nsample, size = alpha2, prob = 1 - 1/( beta2/an + 1))
#       #samples = samples[samples<rarethresh*an]
#       like1 = sum(dnbinom(samples, size = alpha1, prob = 1 - 1/( beta1/an + 1), log = T))
#       like2 = sum(dnbinom(samples, size = alpha2, prob = 1 - 1/( beta2/an + 1), log = T))
#       like_diff = rbind(like_diff, data.table(nsample = nsample, diff = like2 - like1, s1 = s2, s2 = s1))
#     }
#   }
# }

# ggplot(like_diff, aes(x = factor(log10(nsample)), y = pmax(pmin(diff/nsample, 20), -10))) + theme_minimal(base_size = 15) +
#   geom_hline(yintercept = 0, linetype = "dashed", color = "grey") +
#   geom_violin(color = "darkblue", fill = "darkblue") +
#   xlab(expression(log[10]~n)) + 
#   ylab(expression(log(L[1]/L[2])/n)) +
#   facet_wrap(~ s1 + s2, nrow = 1, labeller = "label_both") +
#   ylim(-10, 20)
#   
# 
# ggsave(paste0("comp0.1_like_", an, "_", mu, ".eps"), width = 12, height = 6)
}



