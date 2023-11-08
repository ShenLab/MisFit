import pandas as pd
import os, sys
import json
from scipy.stats import spearmanr

def main(prefix_dir):
	label_dir = "~/data/variant/label/clinvar_int/var_by_uniprot_clinvar/"
	function_dir = "~/data/variant/functional/scores/"
	summary = pd.read_table(os.path.expanduser(f"{label_dir}/../summary_clinvar.txt"))
	all_df = []
	for _, row in summary.iterrows():
		uniprot_id = row['UniprotID']
		model = os.path.expanduser(f"{prefix_dir}/combined_scores/{uniprot_id}.txt.gz")
		if not os.path.exists(model):
			continue
		model_df = pd.read_table(model)
		model_df = model_df.rename(columns = {'Protein_position': 'Uniprot_position'})
		var_df = pd.read_table(os.path.expanduser(f"{label_dir}/{uniprot_id}.txt.gz"))
		var_df = pd.merge(var_df, model_df)
		function_df = pd.read_table(os.path.expanduser(f"{function_dir}/{uniprot_id}_scores.txt.gz"))
		function_df = pd.merge(var_df, function_df)
		all_df.append(function_df)
	df = pd.concat(all_df, axis = 0)
	if not os.path.exists(os.path.expanduser(f"{prefix_dir}/clinvar/")):
		os.makedirs(os.path.expanduser(f"{prefix_dir}/clinvar/"))
	df.to_csv(os.path.expanduser(f"{prefix_dir}/clinvar/clinvar_2_merged.txt.gz"), sep = "\t", index = False)


if __name__=="__main__":
	main(sys.argv[1])
