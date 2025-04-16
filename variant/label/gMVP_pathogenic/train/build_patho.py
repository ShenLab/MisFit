import numpy as np
import json
import pandas as pd
import os

setting_file = open("../../dataset/setting.json", "r")
setting = json.load(setting_file)
setting_file.close()

AA_table = setting["AA_table"]
A = len(AA_table)

def positional_mapping(transcript_id):
    filename = "../../pep/ensembl_to_uniprot_pos/" + transcript_id + ".json"
    if not os.path.exists(filename):
        return None
    with open(filename, "r") as f:
        dictionary = json.load(f)
    return dictionary['mapping']

def _gen_patho(target_df, length):
	target = np.zeros(shape = (length, A))
	train_mask = np.zeros(shape = (length, A))
	val_mask = np.zeros(shape = (length, A))
	for _, row in target_df.iterrows():
		if row['AA_alt'] in AA_table:
			target[row['Uniprot_pos'] - 1][AA_table[row['AA_alt']]] = row['target']
			if row['training'] == 1:
				train_mask[row['Uniprot_pos'] - 1][AA_table[row['AA_alt']]] = 1.
			else:
				val_mask[row['Uniprot_pos'] - 1][AA_table[row['AA_alt']]] = 1.
	return target.astype(np.float32), train_mask.astype(np.float32), val_mask.astype(np.float32)

def main():
	df = pd.read_table("../../list/geneset_uniprot_len.txt")
	output_dir = "var_set/"
	summary  = {'UniprotID':[], 'train_positive': [], 'train_negative': [], 'val_positive': [], 'val_negative': []}
	for _, row in df.iterrows():
		symbol = row['Symbol']
		uniprot_id = row['UniprotID']
		transcript_id = row['TranscriptID']
		length = row['Length']
		mutfile = "var_by_symbol/" + symbol + ".txt"
		funcfile = "../../functional/scores/" + transcript_id + "_scores.txt"
		if os.path.exists(mutfile) and os.path.exists(funcfile):
			mut_df = pd.read_table(mutfile, header = None, names = ["Chrom", "Pos", "Ref", "Alt", "target", "training"])
			func_df = pd.read_table(funcfile)
		else:
			continue
		target_df = pd.merge(mut_df, func_df[['Chrom', 'Pos', 'Ref', 'Alt', 'Protein_position', 'AA_ref', 'AA_alt']])
		pos_mapping = positional_mapping(transcript_id)
		if pos_mapping is None:
			target_df['Uniprot_pos'] = target_df['Protein_position']
		else:
			target_df['Uniprot_pos'] = target_df['Protein_position'].astype('str').map(pos_mapping)
			target_df = target_df.dropna()
		target_df = target_df[(target_df['target']==0)|(target_df['target']==1)]
		if len(target_df) == 0:
			continue
		target_df['Uniprot_pos'] = target_df['Uniprot_pos'].astype('int')
		target, train_mask, val_mask = _gen_patho(target_df, length)
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
