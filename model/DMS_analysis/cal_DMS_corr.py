import pandas as pd
import os, sys
import json
from scipy.stats import spearmanr

def main(prefix_dir):
	DMS_dir = "~/data/DMS/DMS_tables/"
	score_names = ["model_damage", "model_selection", "ESM-1b", "ESM-2", "AlphaMissense", "gMVP", "EVE", "CADD_phred", "REVEL", "PrimateAI", "MPC"]
	exclude_list = ['DDX3X', 'BRCA2']
	result = {'UniprotID': [], 'DMS':[], 'num_AA_change': []}
	for score_name in score_names:
		result[score_name] = []
	summary = pd.read_table(os.path.expanduser(f"{DMS_dir}/summary.txt"))
	summary = summary[~summary['Symbol'].isin(exclude_list)]
	for _, row in summary.iterrows():
		uniprot_id = row['UniprotID']
		DMS_merged = os.path.expanduser(f"{prefix_dir}/{row['DMS']}_merged.txt.gz")
		if not os.path.exists(DMS_merged):
			continue
		DMS_df = pd.read_table(DMS_merged)
		result['UniprotID'].append(uniprot_id)
		result['DMS'].append(row['DMS'])
		score_names_anal = []
		for score_name in score_names:
			if score_name in DMS_df.columns:
				if DMS_df[score_name].count() > (0.2 * DMS_df.shape[0]):
					score_names_anal.append(score_name)
		DMS_df = DMS_df.dropna(subset = score_names_anal)
		DMS_df['DMS'] = row['DMS']
		DMS_df['UniprotID'] = uniprot_id
		for score_name in score_names:
			if score_name in score_names_anal:
				r, p = spearmanr(DMS_df[score_name], DMS_df['functional_rankscore'])
			else:
				r = pd.NA
			result[score_name].append(r)
		result['num_AA_change'].append(len(DMS_df))
	result_df = pd.DataFrame(result)
	result_df = pd.merge(result_df, summary)
	result_df.to_csv(os.path.expanduser(f"{prefix_dir}/correlation_summary.txt"), sep = "\t", index = False)

if __name__=="__main__":
	main(sys.argv[1])
