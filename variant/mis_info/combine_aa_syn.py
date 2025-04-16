import pandas as pd
import os
import re
import gzip
import json

def positional_mapping(transcript_id):
	filename = "../pep/ensembl_to_uniprot_pos/" + transcript_id + ".json"
	if not os.path.exists(filename):
		return None
	with open(filename, "r") as f:
		dictionary = json.load(f)
	return dictionary['mapping']

def main():
	geneset = pd.read_table("../list/geneset_uniprot.txt")
	
	for _, row in geneset.iterrows():
		transcript_name = row['TranscriptID']
		uniprot_id = row['UniprotID']
		if not os.path.exists("syn_info_by_protein/" + transcript_name + "_info.txt"):
			continue
		df = pd.read_table("syn_info_by_protein/" + transcript_name + "_info.txt", header = None, names = ['chrom', 'pos', 'ref', 'alt', 'ensembl_pep_pos', 'ref_aa', 'alt_aa', 'gnomad_mu', 'UKBB_AN', 'UKBB_AC'])
		df = df.dropna()
		AC = df[['ref_aa', 'alt_aa', 'ensembl_pep_pos', 'UKBB_AC', 'gnomad_mu']].groupby(['ref_aa', 'alt_aa', 'ensembl_pep_pos']).sum().reset_index()
		AN = df[['ref_aa', 'alt_aa', 'ensembl_pep_pos', 'UKBB_AN']].groupby(['ref_aa', 'alt_aa', 'ensembl_pep_pos']).max().reset_index()
		df_combine = pd.merge(AN, AC)
		df_combine['UKBB_AC'] = df[['UKBB_AC','UKBB_AN']].min(axis=1)
		pos_mapping = positional_mapping(transcript_name)
		if pos_mapping is None:
			df_combine['uniprot_pep_pos'] = df_combine['ensembl_pep_pos']
		else:
			df_combine['uniprot_pep_pos'] = df_combine['ensembl_pep_pos'].astype('str').map(pos_mapping)
		df_combine = df_combine.dropna().sort_values(by=['uniprot_pep_pos', "alt_aa"])
		df_combine['uniprot_pep_pos'] = df_combine['uniprot_pep_pos'].astype('int')
		if len(df_combine) > 0:
			df_combine.to_csv("syn_info_uniprot_ukbb/" + uniprot_id + "_uniaa.txt.gz", float_format = "%.4e", sep = "\t", index = False)

if __name__=="__main__":
	main()
