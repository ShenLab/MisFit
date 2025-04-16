import numpy as np
import json
import pandas as pd
import os

setting_file = open("setting.json", "r")
setting = json.load(setting_file)
setting_file.close()

AA_table = setting["AA_table"]
A = len(AA_table)

def _gen_pop(uniprot_id, length, popnames, folders):
	filenames = []
	for folder in folders:
		filename = f"{folder}/{uniprot_id}_info.txt.gz"
		if os.path.exists(filename):
			filenames.append(filename)

	AN = np.zeros(shape = (length, A))
	AC = np.zeros(shape = (length, A))
	AA_mask = np.zeros(shape = (length, A))
	if len(filenames) > 0:
		data_list = []
		for filename in filenames:
			data = pd.read_table(filename)
			data_list.append(data)
		data = pd.concat(data_list, ignore_index = True)
	else:
		return AN.astype(np.float32), AC.astype(np.float32), AA_mask.astype(np.float32)
	for _, row in data.iterrows():
		if row['outlier'] or row['LCR']:
			continue
		if row['AA_alt'] in AA_table:
			curr_AN = 0
			curr_AC = 0
			for popname in popnames:
				curr_AN += row[popname + "_AN"]
				curr_AC += row[popname + "_AC"]
			if curr_AN > 0:
				AA_mask[row['Uniprot_AA_pos'] - 1][AA_table[row['AA_alt']]] = 1.
				AN[row['Uniprot_AA_pos'] - 1][AA_table[row['AA_alt']]] = max(curr_AN, AN[row['Uniprot_AA_pos'] - 1][AA_table[row['AA_alt']]])
				AC[row['Uniprot_AA_pos'] - 1][AA_table[row['AA_alt']]] += curr_AC
	AC = np.clip(AC, 0, AN)
	return AN.astype(np.float32), AC.astype(np.float32), AA_mask.astype(np.float32)

def main():
	summary = open("combine_covered_list.txt", "w")
	df = pd.read_table("geneset_uniprot_len.txt")
	af_dir = "pop_ukbb_gnomad/"
	popnames = ["UKBB", "gnomAD_NFE_exome", "gnomAD_NFE_genome"]
	for _, row in df.iterrows():
		uniprot_id = row['UniprotID']
		length = row['Length']
		AN, AC, AA_mask = _gen_pop(uniprot_id, length, popnames, folders = ['mis_info_by_protein_mapping', 'syn_info_by_protein_mapping'])
		np.save(af_dir + uniprot_id + "_AN.npy", AN)
		np.save(af_dir + uniprot_id + "_AC.npy", AC)
		np.save(af_dir + uniprot_id + "_AA_mask.npy", AA_mask)
		if np.sum(AA_mask) > 0:
			print(uniprot_id, file = summary)
	summary.close()

if __name__=="__main__":
	main()
