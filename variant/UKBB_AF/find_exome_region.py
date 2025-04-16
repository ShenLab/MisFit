import pyranges as pr
import pandas as pd
import re

def rename_chrom(chrom):
	chr_search = re.search(r"^chr(\w+)", chrom)
	if chr_search:
		new_chrom = chr_search.group(1)
		if new_chrom == "M":
			return "MT"
		else:
			return new_chrom
	return chrom

def main():
	gr = pr.read_gff3("../list/gencode.v38.basic.annotation.gff3.gz")
	exome = gr[gr.Feature=="exon"]
	exome = exome[['Chromosome','Start','End']]
	exome = exome.unstrand().merge()
	exome.Start = exome.Start - 10
	exome.End = exome.End + 10
	exome = exome.merge()
	exome.Chromosome = exome.Chromosome.apply(rename_chrom)
	exome = exome.sort()
	exome.to_csv("exome_regions.txt", sep = "\t", header = False)

if __name__=="__main__":
	main()

