import pandas as pd
import os, sys
import json

def main(prefix_dir):
	var_df_all = pd.DataFrame()
	var_df = pd.read_table(os.path.expanduser("~/data/variant/ultrarare/mis_ultrarare.txt.gz"))
	for uniprot_id in var_df['UniprotID'].unique():
		var_df_sub = var_df[var_df['UniprotID'] == uniprot_id]
		model = os.path.expanduser(f"{prefix_dir}/combined_scores/{uniprot_id}.txt.gz")
		if not os.path.exists(model):
			continue
		model_df = pd.read_table(model)
		model_df['Uniprot_AA_pos'] = model_df['Protein_position']
		model_df = pd.merge(model_df[['Uniprot_AA_pos', 'AA_alt', 'model_selection']], var_df_sub)
		var_df_all = pd.concat([var_df_all, model_df])

	if not os.path.exists(os.path.expanduser(f"{prefix_dir}/ultrarare/")):
		os.makedirs(os.path.expanduser(f"{prefix_dir}/ultrarare/"))
		
	var_df_all.to_csv(os.path.expanduser(f"{prefix_dir}/ultrarare/mis_ultrarare_merged.txt.gz"), sep = "\t", index = False)


if __name__=="__main__":
	main(sys.argv[1])
