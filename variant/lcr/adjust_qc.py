import pandas as pd
import numpy as np
import re
from os.path import exists
import sys

def main():
	geneset = pd.read_table("geneset_uniprot_len.txt")
	dirs = ['mis_info_by_protein_mapping', 'syn_info_by_protein_mapping', 'lof_info_by_protein']	
	popnames = ['gnomAD_NFE', 'gnomAD_AFR']
	for _, row in geneset.iterrows():
		uniprot_id = row['UniprotID']
		transcript_id = row['TranscriptID']
		for folder in dirs:
			filename = f"{folder}/{uniprot_id}_info.txt.gz"
			if exists(f"/neumann/yz3419/variant/snv_info_2/{filename}"):
				df = pd.read_table(f"/neumann/yz3419/variant/snv_info_2/{filename}")
				if 'Uniprot_AA_pos' in df.columns.unique():
					df['Uniprot_AA_pos'] = df['Uniprot_AA_pos'].astype(int)
				mask_filename = f"snv_qc/{transcript_id}.txt"
				if exists(mask_filename):
					mask_df = pd.read_table(mask_filename, names = ['Chrom', 'Pos', 'Ref', 'Alt', 'Filter'])
					df = pd.merge(df, mask_df, how = "left")
				else:
					df['Filter'] = pd.NA
				for popname in popnames:
					df.loc[df['Filter']=="NOGENOME", [popname + "_genome_AN", popname + "_genome_AC"]] = 0
					df.loc[df['Filter']=="NOEXOME", [popname + "_exome_AN", popname + "_exome_AC"]] = 0
				df.to_csv(filename, index = False, sep = "\t")
if __name__=="__main__":
	main()
