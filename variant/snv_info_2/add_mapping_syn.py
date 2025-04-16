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
		if not os.path.exists("syn_info_by_protein/" + uniprot_id + "_info.txt.gz"):
			continue
		df = pd.read_table("syn_info_by_protein/" + uniprot_id + "_info.txt.gz")
		pos_mapping = positional_mapping(transcript_name)
		if pos_mapping is None:
			df['Uniprot_AA_pos'] = df['Transcript_AA_pos']
		else:
			df['Uniprot_AA_pos'] = df['Transcript_AA_pos'].astype('str').map(pos_mapping)
		df = df.dropna()
		df['Uniprot_AA_pos'] = df['Uniprot_AA_pos'].astype('int')
		df = df.sort_values(by=['Uniprot_AA_pos', "AA_alt"])
		if len(df) > 0:
			df.to_csv("syn_info_by_protein_mapping/" + uniprot_id + "_info.txt", float_format = "%.4e", sep = "\t", index = False)

if __name__=="__main__":
	main()
