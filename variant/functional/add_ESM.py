import pandas as pd
from os.path import exists
import json

def main():
	geneset = pd.read_table("../list/geneset_uniprot.txt")
	ESM_dir = "../ESM-1b/logits_merged/"
	for _, row in geneset.iterrows():
		uniprot_id = row['UniprotID']
		ESM_file = f"{ESM_dir}/{uniprot_id}.txt.gz"
		scores_file = f"scores2/{uniprot_id}_scores.txt.gz"
		ESM_df = pd.read_table(ESM_file)
		ESM_df.columns = ["Uniprot_position", "AA_alt", "ESM-1b"]
		scores_df = pd.read_table(scores_file)
		merged_df = pd.merge(scores_df, ESM_df, how = "left")
		merged_df.to_csv(f"scores/{uniprot_id}_scores.txt.gz", index = False, sep = "\t")

if __name__=="__main__":
	main()

