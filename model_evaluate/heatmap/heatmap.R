library(cowplot)
library(data.table)
library(ggplot2)

# PTEN
prot_start = 14
prot_end = 185
combined_score = fread("P60484.txt.gz")
combined_score = combined_score[(Protein_position >= prot_start) & (Protein_position <= prot_end)]
p1 = ggplot(combined_score, aes(x = Protein_position, y = AA_alt, fill = rank(model_selection))) +
  geom_tile() + 
  theme_nature() +
  theme(panel.grid = element_blank(), panel.background = element_blank()) + 
  scale_fill_gradient(name = element_blank(), low = "white", high = "red") +
  scale_x_continuous(name = element_blank(), expand = c(0,0), limits = c(prot_start, prot_end)) +
  scale_y_discrete(name = "amino acid") + theme(legend.position = "none",
                                                plot.margin = margin(t = 0,
                                                                     r = 0,
                                                                     b = -5,
                                                                     l = 0))
#  annotate("point", color = "black", x = 124, y = 0, shape = 17, size = 3)
#p2 = ggplot(combined_score, aes(x = Protein_position)) + theme_nothing() +
#  xlim(c(prot_start, prot_end)) + annotate("point", color = "black", x = 124, y = 0, shape = 17, size = 3)

AF_data = fread('P60484_info.txt.gz')
chr = paste0("chr", unique(AF_data$Chrom))
genome_start = min(AF_data$Pos)
genome_end = max(AF_data$Pos)

regional_constraint = fread("constraint_z_genome_1kb.raw.download.txt")
regional_constraint = regional_constraint[(chrom==chr)&!((start>genome_end)|(end<genome_start))]
for (i in 1:nrow(regional_constraint)) {
  start = regional_constraint[i, start]
  end = regional_constraint[i, end]
  oe = regional_constraint[i, oe]
  AF_data[(Pos>start)&(Pos<=end), regional_oe:=oe]
}

AF_mis = AF_data[UKBB_AC + gnomAD_NFE_genome_AC + gnomAD_NFE_exome_AC > 0, .(Uniprot_AA_pos)]
AF_data = AF_data[,.(Uniprot_AA_pos, regional_oe)]
AF_data = unique(AF_data, by = "Uniprot_AA_pos")
colnames(AF_data)[1] = "Protein_position"
p3 = ggplot(AF_data[(Protein_position >= prot_start) & (Protein_position <= prot_end)], aes(x = Protein_position, y = regional_oe)) + 
  geom_line() + theme_nature() +
  theme(panel.grid = element_blank(), panel.background = element_blank(), 
        panel.border = element_blank(), axis.text.x = element_blank(),
        plot.margin = margin(t = 0,
                             r = 0,
                             b = 0,
                             l = 0)) + 
  scale_x_continuous(name = "protein position", expand = c(0, 0), limits = c(prot_start, prot_end)) +
  scale_y_continuous(name = "regional o/e")

conservation = fread("P60484_conservation.txt")
p4 = ggplot(conservation[(Protein_position >= prot_start) & (Protein_position <= prot_end)], 
            aes(x = Protein_position, y = entropy)) +
  theme_nature() + 
  theme(panel.grid = element_blank(), panel.background = element_blank(), 
        panel.border = element_blank(), axis.text.x = element_blank(),
        plot.margin = margin(t = -5,
                             r = 0,
                             b = -5,
                             l = 0)) +
  scale_x_continuous(name = element_blank(), expand = c(0,0), limits = c(prot_start, prot_end)) +
  geom_bar(stat="identity")

# struct = fread("P60484_struct_table.txt")
# struct = struct[end-beg>=1]
# struct[, struct:=factor(type, 
#                         levels = c("BEND", "HELX_LH_PP_P", "HELX_RH_3T_P", "HELX_RH_AL_P", "HELX_RH_PI_P", "STRN", "TURN_TY1_P"),
#                         labels = c("bend", 
#                                    "left-handed polyproline helix",
#                                    "right-handed 3-10 helix",
#                                    "right-handed alpha helix",
#                                    "right-handed pi helix",
#                                    "beta strand",
#                                    "type I prime turn"
#                                    )
#                         )]

struct = fread("P60484_uniprot_struct_table.txt")

p2 = ggplot(struct) +
  theme_nature() +
  geom_rect(aes(xmin = start-0.5, xmax = end+0.5, ymin = -1, ymax = 1, fill = structure)) +
  geom_point(color = "black", x = 124, y = 0, shape = 17, size = 2) +
  theme(panel.grid = element_blank(), panel.background = element_blank(), panel.border = element_blank(), 
        axis.text = element_blank(), axis.line = element_blank(), axis.ticks = element_blank(),
        legend.position = "top",
        plot.margin = margin(t = 0,
                             r = 0,
                             b = -30,
                             l = 0)) +
  scale_x_continuous(expand = c(0, 0), limits = c(prot_start, prot_end)) +
  coord_fixed(ratio = 2)

p5 = ggplot(AF_mis) + theme_nature() + 
  geom_density(aes(x = Uniprot_AA_pos), bw = 1) +
  theme(panel.grid = element_blank(), panel.background = element_blank(), 
        panel.border = element_blank(), axis.text.x = element_blank(), 
        plot.margin = margin(t = 0,
                             r = 0,
                             b = 0,
                             l = 0)) +
  scale_x_continuous(name = element_blank(), expand = c(0,0), limits = c(prot_start, prot_end)) +
  scale_y_continuous(name = "density", breaks = c(0))
  

plot_grid(plotlist=list(p2, p1, p4, p5, p3), ncol=1, align='v', axis = 'lr', rel_heights = c(1, 1, 0.5, 0.5, 0.5))

save_nature("heatmap.pdf", width = "double", hw_ratio = 1/2)
