import numpy as np
import json
import pandas as pd
import os

setting_file = open("../../dataset/setting.json", "r")
setting = json.load(setting_file)
setting_file.close()

AA_table = setting["AA_table"]
A = len(AA_table)

def _gen_label(target_df, length):
	target = np.zeros(shape = (length, A))
	train_mask = np.zeros(shape = (length, A))
	val_mask = np.zeros(shape = (length, A))
	for _, row in target_df.iterrows():
		if row['AA_alt'] in AA_table:
			target[row['Uniprot_AA_pos'] - 1][AA_table[row['AA_alt']]] = row['target']
			if row['training'] == 1:
				train_mask[row['Uniprot_AA_pos'] - 1][AA_table[row['AA_alt']]] = 1.
			else:
				val_mask[row['Uniprot_AA_pos'] - 1][AA_table[row['AA_alt']]] = 1.
	return target.astype(np.float32), train_mask.astype(np.float32), val_mask.astype(np.float32)

def main():
	df = pd.read_table("../../list/geneset_uniprot_len.txt")
	output_dir = "var_set/"
	summary  = {'UniprotID':[], 'train_positive': [], 'train_negative': [], 'val_positive': [], 'val_negative': []}
	for _, row in df.iterrows():
		uniprot_id = row['UniprotID']
		length = row['Length']
		mutfile = "var_by_uniprot/" + uniprot_id + ".txt"
		if os.path.exists(mutfile):
			target_df = pd.read_table(mutfile, header = None, names = ["UniprotID", "AA_ref", "Uniprot_AA_pos", "AA_alt", "target", "training"])
		else:
			continue
		target_df = target_df[(target_df['target']==0)|(target_df['target']==1)]
		if len(target_df) == 0:
			continue
		target, train_mask, val_mask = _gen_label(target_df, length)
		summary['UniprotID'].append(uniprot_id)
		summary['train_positive'].append(sum((target_df['target']==1) & (target_df['training']==1)))
		summary['train_negative'].append(sum((target_df['target']==0) & (target_df['training']==1)))
		summary['val_positive'].append(sum((target_df['target']==1) & (target_df['training']==0)))
		summary['val_negative'].append(sum((target_df['target']==0) & (target_df['training']==0)))
		np.save(output_dir + uniprot_id + "_target.npy", target)
		np.save(output_dir + uniprot_id + "_train_mask.npy", train_mask)
		np.save(output_dir + uniprot_id + "_val_mask.npy", val_mask)

	summary = pd.DataFrame(summary)
	summary.to_csv("summary.txt", sep = "\t", index = False)
if __name__=="__main__":
	main()
