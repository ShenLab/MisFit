import pandas as pd
from os.path import exists
import json

def main():
	geneset = pd.read_table("../list/geneset_uniprot.txt")
	EVE_dir = "../EVE/EVE_scores/"
	mapping_dir = "../pep/uniprot_to_ensembl_pos/"
	for _, row in geneset.iterrows():
		uniprot_id = row['UniprotID']
		transcript_id = row['TranscriptID']
		EVE_file = f"{EVE_dir}/{uniprot_id}.txt.gz"
		scores_file = f"scores/{transcript_id}_scores.txt.gz"
		if exists(EVE_file) and exists(scores_file):
			EVE_df = pd.read_table(EVE_file)
			EVE_df.columns = ["AA_ref", "Protein_position", "AA_alt", "EVE_raw", "EVE", "EVE_class"]
			mapping_file = f"{mapping_dir}/{uniprot_id}.json"
			if exists(mapping_file):
				with open(mapping_file, "r") as f:
					mapping = json.load(f)
				pos = EVE_df['Protein_position'].map(mapping['mapping'])
				EVE_df['Protein_position'] = pos
				EVE_df = EVE_df.dropna()
			scores_df = pd.read_table(scores_file)
			original_columns = scores_df.columns
			scores_df.drop(columns = "EVE", inplace = True)
			merged_df = pd.merge(scores_df, EVE_df[["Protein_position", "AA_alt", "EVE"]], how = "left")
			merged_df = merged_df[original_columns]
			merged_df.to_csv(f"scores2/{transcript_id}_scores.txt.gz", index = False, sep = "\t")

if __name__=="__main__":
	main()

