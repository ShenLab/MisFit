import pandas as pd
import gzip
import re

def main():
	geneset = pd.read_table("summary_clinvar.txt")
	geneset_all = pd.read_table("../../list/geneset_uniprot_len.txt")
	geneset_all = geneset_all[geneset_all['UniprotID'].isin(geneset['UniprotID'])]
	geneset_dict = {}
	for _, row in geneset_all.iterrows():
		geneset_dict[row['Symbol']] = row['UniprotID']
	var_file = gzip.open("clinvar_compare_snv_annot_vus.vcf.gz", "rt")
	output_file = gzip.open("clinvar_vus.txt.gz", "wt")
	print("Chrom\tPos\tRef\tAlt\tUniprotID\tSymbol", file = output_file)
	for line in var_file:
		if re.search(r"^#", line):
			continue
		line_split = line.strip().split("\t")
		gene_search = re.search(r"GENEINFO=(\w+)", line_split[7])
		if gene_search:
			gene = gene_search.group(1)
			if gene in geneset_dict:
				print(line_split[0], line_split[1], line_split[3], line_split[4], geneset_dict[gene], gene, sep = "\t", file = output_file)
	var_file.close()
	output_file.close()

if __name__=="__main__":
	main()
