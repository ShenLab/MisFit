import pandas as pd
from os.path import exists
import json

def main():
	geneset = pd.read_table("../list/geneset_uniprot.txt")
	for _, row in geneset.iterrows():
		uniprot_id = row['UniprotID']
		alpha_file = f"alphamissense/{uniprot_id}.txt"
		scores_file = f"scores/{uniprot_id}_scores.txt.gz"
		scores_df = pd.read_table(scores_file)
		scores_df['Uniprot_position'] = scores_df['Uniprot_position'].fillna(-1)
		scores_df['Uniprot_position'] = scores_df['Uniprot_position'].astype(int)
		if not exists(alpha_file):
			scores_df['AlphaMissense'] = pd.NA
			scores_df.to_csv(f"scores2/{uniprot_id}_scores.txt.gz", index = False, sep = "\t")
		else:
			alpha_df = pd.read_table(alpha_file, names = ['Pos', 'Alt', 'AlphaMissense'])
			merged_df = pd.merge(scores_df, alpha_df, how = "left")
			merged_df.to_csv(f"scores2/{uniprot_id}_scores.txt.gz", index = False, sep = "\t")

if __name__=="__main__":
	main()

