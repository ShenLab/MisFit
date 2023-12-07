library(data.table)
library(ggplot2)
source("../../../settings/plot_settings.R")

ngroup = 10
data_p = data.table()
for (model_index in c(0, 1, 2, 3)) {
  data = fread(paste0("ultrarare_model_", model_index, ".txt.gz"))
  #data = data[AC0==0]
  data = data[(AC0/AN0) < 5e-6]
  data[, rank:=as.integer((rank(model_selection)-0.1)/nrow(data) * ngroup)+1]
  data_p0 = data[, mean(AC1==0), by = rank]
  #data_p0 = data[, mean(AC1/AN1>5e-5), by = rank]
  colnames(data_p0)[2] = 'probability'
  data_p0[, model_index:=model_index]
  data_p = rbind(data_p, data_p0)
}
p1 = ggplot(data_p, aes(x = rank, y = probability, color = factor(model_index))) + geom_line(linewidth = 0.8) +
  theme_nature() + 
  scale_color_discrete(name = element_blank(), labels = c("model 0", 
                                                  "model 1",
                                                  "model 2",
                                                  "MisFit")) +
  scale_x_continuous(name = "selection decile", breaks = seq(1, 10), minor_breaks = NULL) +
  scale_y_continuous(name = "P(AFR AF = 0)", breaks = seq(0, 1, 0.1))

data_p= data.table()
for (model_index in c(0, 1, 2, 3)) {
  data = fread(paste0("ultrarare_model_", model_index, ".txt.gz"))
  #data = data[AC0==0]
  data = data[(AC0/AN0) < 5e-6]
  data[, rank:=as.integer((rank(model_selection)-0.1)/nrow(data) * ngroup)+1]
  #data_p0 = data[, mean(AC1==0), by = rank]
  data_p0 = data[, mean(AC1/AN1>5e-5), by = rank]
  colnames(data_p0)[2] = 'probability'
  data_p0[, model_index:=model_index]
  data_p = rbind(data_p, data_p0)
}
p2 = ggplot(data_p, aes(x = rank, y = probability, color = factor(model_index))) + geom_line(linewidth = 0.8) +
  theme_nature() + 
  scale_color_discrete(name = element_blank(), labels = c("model 0", 
                                                          "model 1",
                                                          "model 2",
                                                          "MisFit")) +
  scale_x_continuous(name = "selection decile", breaks = seq(1, 10), minor_breaks = NULL) +
  scale_y_continuous(name = "P(AFR AF > 5e-5)", breaks = seq(0, 1, 0.02))
#  theme(legend.position = c(0.3, 0.8), legend.direction="vertical")

data_s = data.table()
for (model_index in c(0, 1, 2, 3)) {
  data = fread(paste0("ultrarare_model_", model_index, ".txt.gz"))
  data = data[(AC0/AN0) < 5e-6]
  data[, s:=sigmoid(model_selection)]
  data[, model_index:=model_index]
  data_s = rbind(data_s, data[, .(s, model_index)])
}
p3 = ggplot(data_s, aes(x = s, color = factor(model_index))) + geom_density(bw = 0.1, linewidth = 0.8) +
  theme_nature() +
  scale_x_continuous(trans = "log10", labels = trans_format("log10", math_format(10^.x))) +
  scale_color_discrete(name = element_blank(), labels = c("model 0", 
                                                          "model 1",
                                                          "model 2",
                                                          "MisFit"))

arrange_nature(arrange_nature(p1, p2, common.legend = T, legend = "bottom"), 
               p3 + theme(legend.position = "none"), nrow = 2, labels = c("", "c"), heights = c(2, 1))
save_nature("proportion.pdf", hw_ratio = 3/4)
