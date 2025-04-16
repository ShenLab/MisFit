import numpy as np
import json
import pandas as pd
import os

setting_file = open("setting.json", "r")
setting = json.load(setting_file)
setting_file.close()

AA_table = setting["AA_table"]
A = len(AA_table)

def _gen_pop(uniprot_id, length):
	filename = "syn_info_uniprot_ukbb/" + uniprot_id + "_uniaa.txt.gz"
	# mu = np.zeros(shape = (length, A))
	# AN = np.zeros(shape = (length, A))
	# AC = np.zeros(shape = (length, A))
	# AA_mask = np.zeros(shape = (length, A))
	mu = np.load("pop_ukbb/" + uniprot_id + "_mu.npy")
	AN = np.load("pop_ukbb/" + uniprot_id + "_AN.npy")
	AC = np.load("pop_ukbb/" + uniprot_id + "_AC.npy")
	AA_mask = np.load("pop_ukbb/" + uniprot_id + "_AA_mask.npy")

	if os.path.exists(filename):
		data = pd.read_table(filename)
	else:
		return mu.astype(np.float32), AN.astype(np.float32), AC.astype(np.float32), AA_mask.astype(np.float32)
	for _, row in data.iterrows():
		if row['alt_aa'] in AA_table:
			mu[row['uniprot_pep_pos'] - 1][AA_table[row['alt_aa']]] = row['gnomad_mu']
			AN[row['uniprot_pep_pos'] - 1][AA_table[row['alt_aa']]] = row['UKBB_AN']
			AC[row['uniprot_pep_pos'] - 1][AA_table[row['alt_aa']]] = row['UKBB_AC']
			AA_mask[row['uniprot_pep_pos'] - 1][AA_table[row['alt_aa']]] = 1.
	return mu.astype(np.float32), AN.astype(np.float32), AC.astype(np.float32), AA_mask.astype(np.float32)

def main():
	df = pd.read_table("../list/geneset_uniprot_len.txt")
	output_dir = "pop_ukbb_syn/"
	for _, row in df.iterrows():
		uniprot_id = row['UniprotID']
		length = row['Length']
		mu, AN, AC, AA_mask = _gen_pop(uniprot_id, length)
		np.save(output_dir + uniprot_id + "_mu.npy", mu)
		np.save(output_dir + uniprot_id + "_AN.npy", AN)
		np.save(output_dir + uniprot_id + "_AC.npy", AC)
		np.save(output_dir + uniprot_id + "_AA_mask.npy", AA_mask)

if __name__=="__main__":
	main()
