import pandas as pd
import numpy as np
from os.path import exists
import json
with open("setting.json", "r") as f:
	setting = json.load(f)

def main():
	geneset = pd.read_table("../list/geneset_uniprot_len.txt")
	for _, row in geneset.iterrows():
		uniprot_id = row['UniprotID']
		msa = np.load(f"MSA_one/{uniprot_id}_count.npy")
		if exists(f"mis_benign/{uniprot_id}.txt.gz"):
			df = pd.read_table(f"mis_benign/{uniprot_id}.txt.gz")
			df = df.dropna()
			for _, var in df.iterrows():
				pos = var['Uniprot_position']
				aa_alt = var['AA_alt']
				aa_index = setting['AA_table'][aa_alt]
				msa[int(pos)-1][aa_index] = 1
		np.save(f"zoonomia_primate_one/{uniprot_id}_count.npy", msa)

if __name__=="__main__":
	main()
