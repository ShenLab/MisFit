library(data.table)

data = fread("ensembl_104_transcripts.txt")
colnames(data) = c("GeneID", "TranscriptID", "Strand", "basic", "MANE", "canonical", "ProteinID", "Chrom", "Symbol", "HGNC")

all_chrs = c(1:22, "X", "Y", "MT")
data = data[is.element(Chrom, all_chrs)]

MANE_select = (data$MANE != "")
data[, MANE:=MANE_select]
canonical_select = (data$canonical == 1)
canonical_select[is.na(canonical_select)] = F
data[, canonical:=canonical_select]
basic_select = (data$basic=="GENCODE basic")
data[, basic:=basic_select]
strand = c()
strand[data$Strand==1]="+"
strand[data$Strand==-1]="-"
data[, Strand:=strand]

MANE_set = data[MANE==T]
MANE_gene = MANE_set[, GeneID]
MANE_HGNC = MANE_set[, HGNC]

# unique GeneID
canonical_set = data[!is.element(GeneID, MANE_gene)&canonical==T]

select_data = rbind(MANE_set, canonical_set)
select_data = unique(select_data[, .(GeneID, TranscriptID, ProteinID, Symbol, Strand, Chrom)])
select_data = select_data[order(Chrom),]

length(unique(select_data$GeneID))
length(unique(select_data$Symbol))
which(table(select_data$Symbol)>1)

fwrite(select_data, file = "geneset.txt", sep = "\t")

