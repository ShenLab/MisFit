import pandas as pd
import numpy as np
import re
from os.path import exists
import sys

def main():
	mask = pd.read_table("exome_lcr_region.txt.gz")
	geneset = pd.read_table("geneset_uniprot_len.txt")
	dirs = ['mis_info_by_protein_mapping', 'syn_info_by_protein_mapping', 'lof_info_by_protein']	
	for i, row in geneset.iterrows():
		uniprot_id = row['UniprotID']
		gene_id = row['GeneID']
		for folder in dirs:
			filename = f"{folder}/{uniprot_id}_info.txt.gz"
			if exists(filename):
				df = pd.read_table(filename, dtype = {'Chrom': str, 'Filter': str, 'outlier': "boolean"})
				df['LCR'] = False
				mask_gene = mask[mask['GeneID'] == gene_id]
				for _, mask_region in mask_gene.iterrows():
					df.loc[(df['Pos']>=mask_region['Start'])&(df['Pos']<=mask_region['End']), 'LCR'] = True
				df.to_csv(filename, index = False, sep = "\t")
		if (i + 1) % 1000 == 0:
			print(f"{i} processed")
if __name__=="__main__":
	main()
