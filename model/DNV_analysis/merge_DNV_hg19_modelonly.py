import pandas as pd
import os, sys
import json
from scipy.stats import spearmanr

def main(prefix_dir, exist_dir, model_name, condition = "ASD"):
	DNV_df = pd.read_table(os.path.expanduser(f"{exist_dir}/DNV/{condition}_mis_hg19_merged.txt.gz"))
	if model_name in DNV_df.columns:
		DNV_df = DNV_df.drop(columns = [model_name])
	DNV_df2 = pd.DataFrame()
	for uniprot_id in DNV_df['UniprotID'].unique():
		DNV_df_sub = DNV_df[DNV_df['UniprotID']==uniprot_id]
		model = os.path.expanduser(f"{prefix_dir}/combined_scores/{uniprot_id}.txt.gz")
		if not os.path.exists(model):
			continue
		model_df = pd.read_table(model)
		model_df = model_df[['Protein_position', 'AA_alt', 'model_selection']]
		model_df = model_df.rename(columns = {'Protein_position': 'Uniprot_position', 'model_selection': f"{model_name}"})
		model_df = pd.merge(DNV_df_sub, model_df, how = "left")
		DNV_df2 = pd.concat([DNV_df2, model_df])
	if not os.path.exists(os.path.expanduser(f"{prefix_dir}/DNV/")):
		os.makedirs(os.path.expanduser(f"{prefix_dir}/DNV/"))
	
	DNV_df2.to_csv(os.path.expanduser(f"{prefix_dir}/DNV/{condition}_mis_hg19_merged.txt.gz"), sep = "\t", index = False)


if __name__=="__main__":
	main(sys.argv[1], sys.argv[2], sys.argv[3], "ASD")
	main(sys.argv[1], sys.argv[2], sys.argv[3], "NDD")
