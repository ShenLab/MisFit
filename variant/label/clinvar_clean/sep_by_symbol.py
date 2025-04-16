import gzip
import pandas as pd
import re
import os

def main():
	geneset = pd.read_table("../../list/geneset_uniprot.txt")
	gene_dict = {}
	for _,  row in geneset.iterrows():
		gene_dict[row['Symbol']] = row['UniprotID']
	
	input_file = gzip.open("clinvar_compare_snv_annot_clean.vcf.gz", "rt")
	for line in input_file:
		if re.search(r"^#", line):
			continue
		line_split = line.strip().split()
		info = line_split[7]
		symbol = re.search(r"GENEINFO=(\w+)", info).group(1)
		if symbol in gene_dict:
			uniprot_id = gene_dict[symbol]
		else:
			continue
		recent = re.search(r"RECENT=(\d)", info).group(1)
		sig = re.search(r"CLNSIG=([^;]+)", info).group(1)
		if re.search(r"Pathogenic|Likely_pathogenic", sig):
			label = 1
		else:
			label = 0
		output_filename = f"var_by_uniprot/{uniprot_id}_info.txt"
		if not os.path.exists(output_filename):
			output_file = open(output_filename, "w")
			print("Chrom", "Pos", "Ref", "Alt", "UniprotID", "Symbol", "Recent", "Label", sep = "\t", file = output_file)
		else:
			output_file = open(output_filename, "a")
		print(line_split[0], line_split[1], line_split[3], line_split[4], uniprot_id, symbol, recent, label, sep = "\t", file = output_file)
		output_file.close()
	input_file.close()

if __name__=="__main__":
	main()
