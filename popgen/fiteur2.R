library(data.table)
library(ggplot2)

# final effective population size
Ne = 3386437
rarethresh = 0.01
Ns = 2*Ne

dnbinom2 = function(AC, an, alpha, beta) {
  dnbinom(AC, size = alpha, prob = 1 - 1/((beta/an) + 1))
}

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

# get all simulated mean and var
dt = data.table()
for (mu0 in c(5e-7, 5e-8, 5e-9, 5e-10, 1.6e-7, 1.6e-8, 1.6e-9)) {
  data = fread(paste0("sim_q/simulation_", mu0, ".csv"))
  colnames(data) = c("mu", "s", "AC", "Occ")
  data[, AF:=AC/2/Ne]
  for (s0 in c(1.6e-06, 5.0e-06, 1.6e-05, 5.0e-05, 1.6e-04, 5.0e-04, 1.6e-03, 5.0e-03, 1.6e-02, 5.0e-02, 1.6e-01, 5.0e-01, 1)) {
    AC = data[s==s0, rep(AC, Occ)]
    AF = data[s==s0, rep(AF, Occ)]
    k = rhyper(length(AC), AC, 2*Ne-AC, Ns)
    #k = rbinom(length(AF), Ns, AF)
    pAF0 = sum(AF==0)/length(AF)
    pk0 = sum(k==0)/length(k)
    
    dt = rbind(dt, data.table(mu=mu0, s=s0, mean=mean(AF), var=var(AF), 
                              kmean = mean(k), kvar = var(k),
                              pAF0 = pAF0,
                              pk0 = pk0
    ))
  }
}



# transformed s 
dt[, s_trans := logit(s)]

# fit ne = f(s_trans)
fitp0 = function(p0, mu, s, minlogn = 11, maxlogn = 18) {
  logn = seq(minlogn, maxlogn, 0.01)
  alpha = 4 * exp(logn) * mu
  beta = 4 * exp(logn) * s
  like = dnbinom2(0, 2*Ne, alpha, beta)
  like_diff = abs(log(like) - log(p0))
  return(logn[which.min(like_diff)])
}


logne = numeric(nrow(dt))
for (i in 1:nrow(dt)) {
  logne[i] = fitp0(dt[i, pAF0], dt[i, mu], dt[i, s])
}

# for (i in 1:nrow(dt)) {
#   logne[i] = fitpmin(dt[i, pAFmin], dt[i, mu], dt[i, s])
# }
dt[, lognp0:=logne]


ggplot(dt[mu>=1.6e-9&mu<=1.6e-7], aes(x = log10(s), y = (lognp0 )/log(10), color = factor(mu))) + geom_line() + 
  theme_bw(base_size = 14) +
  ylab("estimated log10(Ne)") +
  scale_color_discrete(name = "mutation rate")
ggsave("est_ne.svg", width = 18, height = 9, units = "cm")


x = dt[mu>=1.6e-9&mu<=1.6e-7&s<1, logit(s)]
y = dt[mu>=1.6e-9&mu<=1.6e-7&s<1, lognp0]

#nlsfitn <- nls(y ~ a * x ^ 3 + -0.5*(3*a*x1) * x ^ 2  + log(Ne), start = list(a = -0.1))
nlsfitn = nls(y ~ a * x ^ 3 + b * x ^ 2 + c * x + d , start  = list(a = 0.1, b = 1, c = 0, d = 10))

a = nlsfitn$m$getPars()['a']
b = nlsfitn$m$getPars()['b']
c = nlsfitn$m$getPars()['c']
d = nlsfitn$m$getPars()['d']
#b = -0.5*(3*a*x1)
#c = -0.5*(4*a*x1^2+3*b*x1)

fitn = function(x) {
  x = pmin(x, 5)
  x = pmax(x, -13.3)
  fitn0  = function(x) {
      return(a * x ^ 3 + b * x ^ 2 + c * x + d)
    }

  return(sapply(x, fitn0))
}

plot(seq(-13, 3), fitn(seq(-13, 3)), "l") + points(x, y)

ne = 1e4
dt[, alpha0 := 4*ne*mu]
dt[, beta0 := 4*ne*s]
dt[, nbp0:=1/( 4*ne*s/Ns + 1)]
ne = 1e5
dt[, alpha1 := 4*ne*mu]
dt[, beta1 := 4*ne*s]
dt[, nbp1:=1/( 4*ne*s/Ns + 1)]
ne = 1e6
dt[, alpha2 := 4*ne*mu]
dt[, beta2 := 4*ne*s]
dt[, nbp2:=1/( 4*ne*s/Ns + 1)]
ne = 1e7
dt[, alpha3 := 4*ne*mu]
dt[, beta3 := 4*ne*s]
dt[, nbp3:=1/( 4*ne*s/Ns + 1)]

dt[, alpha := 4*exp(fitn(s_trans))*mu]
dt[, beta := 4*exp(fitn(s_trans))*s]
dt[, nbp := 1/( 4*exp(fitn(s_trans))*s/Ns + 1)]


ggplot(dt[mu==1.6e-8], ) + theme_bw() +
  theme_bw(base_size = 14) + 
  geom_line(aes(x = log10(s), y = log10(mu/s*Ns)), color = "grey", linetype = "dashed") +
  #geom_line(aes(x = log10(s), y = log10(nbp * alpha /  (1 - nbp))), color = "orange") +
  geom_line(aes(x = log10(s), y = log10(kmean)), color = "black") +
  ylab("log10 (mean of allele counts)")
ggsave("sim_mean_p0_1.6e-8.svg", width = 9, height = 9, units = "cm")

ggplot(dt[mu==1.6e-8], ) + theme_bw() +
  theme_bw(base_size = 14) + 
  geom_line(aes(x = log10(s), y = log10(mu/s*Ns)), color = "grey", linetype = "dashed") +
  geom_line(aes(x = log10(s), y = log10(nbp1 * alpha0 /  (1 - nbp0)^2)), color = "yellow", linetype = "dashed") +
  geom_line(aes(x = log10(s), y = log10(nbp1 * alpha1 /  (1 - nbp1)^2)), color = "orange", linetype = "dashed") +
  geom_line(aes(x = log10(s), y = log10(nbp2 * alpha2 /  (1 - nbp2)^2)), color = "red", linetype = "dashed") +
  geom_line(aes(x = log10(s), y = log10(nbp3 * alpha3 /  (1 - nbp3)^2)), color = "brown", linetype = "dashed") +
  geom_line(aes(x = log10(s), y = log10(nbp * alpha /  (1 - nbp)^2)), color = "blue") +
  geom_line(aes(x = log10(s), y = log10(kvar)), color = "black") +
  ylab("log10 variance of allele counts")
ggsave("sim_var_p0_1.6e-8.svg", width = 9, height = 9, units = "cm")

ggplot(dt, aes(x=log10(s), y = pk0, color = factor(mu))) +
  theme_bw(base_size = 15) +
  xlab("log10 selection coefficient") +
  scale_y_continuous(name = "probability of observing 0 count", breaks = seq(0, 1, 0.2)) +
  scale_color_discrete(name = "mutation rate") +
  geom_point()
ggsave(paste0("pk0_", Ns, ".eps"), width = 6, height = 4)

an = 2000000
likedt = data.table()
mu0 = 1.6e-8
for (s0 in c( 1.6e-5, 1.6e-4, 1.6e-3, 1.6e-2, 1.6e-1)) {
  logne = fitn(logit(s0))
  alpha = mu0 * 4 * exp(logne)
  beta = s0 * 4 * exp(logne)
  ac = seq(0, an*0.01, 1)
  like = dnbinom(ac, size = alpha, prob = 1 - 1/( beta/an + 1))
  likedt = rbind(likedt, data.table(ac, like, s=s0))
}
lims = 0.01
ggplot(likedt[ac/an<=lims], aes(x = ac/an/lims, y = log(like), color = factor(s), group = factor(s))) + geom_line() + 
  theme_bw(base_size = 15) + 
  scale_x_continuous(breaks = seq(0, 1, 0.1), name = paste0("observed allele frequency \u00D7", lims)) +
  ylab("log likelihood") +
  scale_color_discrete(name = "selection coefficient") +
  ggtitle(paste0("allele number = ", an, ", mu = ", mu0)) +
  ylim(-30, 0)
ggsave(paste0("like_", an, "_", mu0, "_", lims, ".eps"), width = 8, height = 6)

