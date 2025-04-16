import pandas as pd
from os.path import exists
import numpy as np

def main():
	df_all = pd.DataFrame()
	geneset = pd.read_table("../dataset/data_15/list_1.txt")
	for uniprot_id in geneset['UniprotID'].unique():
		filename = f"../snv_info_2/mis_info_by_protein_mapping/{uniprot_id}_info.txt.gz"
		if exists(filename):
			df = pd.read_table(filename)
			df['AN0'] = 0
			df['AN1'] = 0
			df['AC0'] = 0
			df['AC1'] = 0
			df['mu'] = df['roulette_mu']
			for popname in ['UKBB', 'gnomAD_NFE_exome', 'gnomAD_NFE_genome']:
				df['AN0'] += df[popname + "_AN"]
				df['AC0'] += df[popname + "_AC"]
			df = df[(df['AN0'] > 4e5) & (df['mu'] > 1e-7)]
			df['AF0'] = df['AC0']/df['AN0']
			df = df[df['AF0'] < 1e-5]
			for popname in ['gnomAD_AFR_exome', 'gnomAD_AFR_genome']:
				df['AN1'] += df[popname + "_AN"]
				df['AC1'] += df[popname + "_AC"]
			df_all = pd.concat([df_all, df[['Chrom', 'Pos', 'Ref', 'Alt', 'UniprotID', 'TranscriptID', 'AA_ref', 'AA_alt', 'Transcript_AA_pos', 'Uniprot_AA_pos', 'mu', 'AN0', 'AC0', 'AN1', 'AC1']]])
	df_all.to_csv("mis_ultrarare.txt.gz", compression = "gzip", index = False, sep = "\t")

if __name__=="__main__":
	main()

