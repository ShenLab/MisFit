import pandas as pd
import re
import gzip
from Bio import SeqIO
from os.path import exists

def get_depth(MSA_file):
	if exists(MSA_file):
		with gzip.open(MSA_file, "rt") as f:
			records = list(SeqIO.parse(f, "fasta"))
		return len(records)
	else:
		return 0

def main():
	geneset_df = pd.read_table("../list/geneset_uniprot.txt")
#	geneset_df = pd.read_table("error.txt")
	align_dir = "MSA_by_uniprot/"
	summary_f = open("genetree_MSA_depth_summary.txt", "w")
	error_f = open("error.txt", "w")
	print("UniprotID", "GeneID", "depth", sep = '\t', file = summary_f)
	print("UniprotID", "GeneID", "depth_0", "depth_1", sep = '\t', file = error_f)
	for i, row in geneset_df.iterrows():
		uniprot_id = row['UniprotID']
		gene_id = row['GeneID']
		MSA_file_0 = "Ensembl_genetree/" + gene_id + "_aligned.fasta.gz"
		MSA_file_1 = "MSA_by_uniprot/" + uniprot_id + "_aligned.fasta.gz"
		depth_0 = get_depth(MSA_file_0)
		depth_1 = get_depth(MSA_file_1)
		
		if (depth_0 == 0) and (depth_1 == 1):
			print(uniprot_id, gene_id, depth_1, sep = '\t', file = summary_f)
		elif (depth_0 != depth_1) or (depth_1 == 0):
			print(uniprot_id, gene_id, depth_0, depth_1, sep = '\t', file = error_f)
		else:
			print(uniprot_id, gene_id, depth_1, sep = '\t', file = summary_f)
	summary_f.close()
	error_f.close()

if __name__=="__main__":
	main()



