import numpy as np
import json
import pandas as pd
import os

setting_file = open("setting.json", "r")
setting = json.load(setting_file)
setting_file.close()

AA_table = setting["AA_table"]
A = len(AA_table)

def _gen_mu(uniprot_id, length, munames, info_dirs):
	mu = np.zeros(shape = (length, A))
	for info_dir in info_dirs:
		filename = f"{info_dir}/{uniprot_id}_info.txt.gz"
		if os.path.exists(filename):
			data = pd.read_table(filename)
			for _, row in data.iterrows():
				for muname in munames:
					if row[muname] > 0:
						mu = row[muname]
						break
				if row['AA_alt'] in AA_table:
					mu[row['Uniprot_AA_pos'] - 1][AA_table[row['AA_alt']]] += mu
		else:
			continue
	return mu.astype(np.float32)

def main():
	df = pd.read_table("../list/geneset_uniprot_len.txt")
	mu_dir = "mu_roulette"
	munames = ["roulette_mu", "gnomAD_mu_old"]
	info_dirs = ["mis_info_by_protein_mapping", "syn_info_by_protein_mapping"]
	for _, row in df.iterrows():
		uniprot_id = row['UniprotID']
		length = row['Length']	
		mu = _gen_mu(uniprot_id, length, munames, info_dirs)
		np.save(f"{mu_dir}/{uniprot_id}_mu.npy", mu)

if __name__=="__main__":
	main()
