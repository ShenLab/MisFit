import pandas as pd
import numpy as np
import os, sys
import json
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

def main(prefix_dir, score_name_exclude = None):
	score_names = ["model_damage", "model_selection", 'ESM-2', 'ESM-1b', 'AlphaMissense', 'EVE', 'PrimateAI', 'gMVP', 'CADD_phred', 'REVEL', 'MPC']
	if score_name_exclude is None:
		suffix = "all"
	else:
		suffix = "exclude"
		for score_name in score_name_exclude:
			score_names.remove(score_name)
	result = {'method': [], 'by': [], 'group':[], 'AUC': [], 'num_gene': [], 'num_var': []}
	label_merged = pd.read_table(os.path.expanduser(f"{prefix_dir}/clinvar_2_merged.txt.gz"))
	# process number of labels
	var_per_gene = label_merged.groupby(['UniprotID']).size().reset_index(name='counts')
	var_per_gene['annot_group'] = pd.NA
	var_per_gene.loc[var_per_gene['counts'] < 5, 'annot_group'] = 0
	var_per_gene.loc[(var_per_gene['counts'] >= 5) & (var_per_gene['counts'] < 20), 'annot_group'] = 1
	var_per_gene.loc[var_per_gene['counts'] >= 20, 'annot_group'] = 2
	label_merged = pd.merge(label_merged, var_per_gene[['UniprotID', 'annot_group']])
	# annot group
	for i in [0, 1, 2]:
		df = label_merged[label_merged['annot_group'] == i][['UniprotID', 'Label'] + score_names].dropna()
		for score_name in score_names:
			AP = roc_auc_score(df['Label'], df[score_name])
			result['num_gene'].append(len(df['UniprotID'].unique()))
			result['num_var'].append(len(df))
			result['method'].append(score_name)
			result['by'].append('label')
			result['group'].append(i)
			result['AUC'].append(AP)
	# process depth
#	MSA_depth_file = "~/data/variant/MSA/genetree_MSA_depth_summary.txt"
	MSA_depth_file = "~/data/variant/MSA/uniref50_size.tsv.gz"
	MSA_depth = pd.read_table(os.path.expanduser(MSA_depth_file), header = 0, names = ['UniprotID', 'Cluter', 'Depth'])
	MSA_depth['depth_group'] = pd.NA
	MSA_depth.loc[MSA_depth['Depth'] < 100, 'depth_group'] = 0
	MSA_depth.loc[(MSA_depth['Depth'] >= 100) & (MSA_depth['Depth'] < 500), 'depth_group'] = 1
	MSA_depth.loc[MSA_depth['Depth'] >= 500 , 'depth_group'] = 2
	label_merged = pd.merge(label_merged, MSA_depth[['UniprotID', 'depth_group']])
	# depth group	
	for i in [0, 1, 2]:
		df = label_merged[label_merged['depth_group'] == i][['UniprotID', 'Label'] + score_names].dropna()
		for score_name in score_names:
			AP = roc_auc_score(df['Label'], df[score_name])
			result['num_gene'].append(len(df['UniprotID'].unique()))
			result['num_var'].append(len(df))
			result['method'].append(score_name)
			result['by'].append("depth")
			result['group'].append(i)
			result['AUC'].append(AP)
	# time group
	for i in [0, 1]:
		df = label_merged[label_merged['Recent'] == i][['UniprotID', 'Label'] + score_names].dropna()
		for score_name in score_names:
			AP = roc_auc_score(df['Label'], df[score_name])
			result['num_gene'].append(len(df['UniprotID'].unique()))
			result['num_var'].append(len(df))
			result['method'].append(score_name)
			result['by'].append("time")
			result['group'].append(i)
			result['AUC'].append(AP)
	# all
	df = label_merged[['UniprotID', 'Label'] + score_names].dropna()
	for score_name in score_names:
		AP = roc_auc_score(df['Label'], df[score_name])
		result['num_gene'].append(len(df['UniprotID'].unique()))
		result['num_var'].append(len(df))
		result['method'].append(score_name)
		result['by'].append("all")
		result['group'].append("all")
		result['AUC'].append(AP)
	result_df = pd.DataFrame(result)

	result_df.to_csv(os.path.expanduser(f"{prefix_dir}/Clinvar_summary_2_{suffix}.txt"), sep = "\t", index = False)


if __name__=="__main__":
	main(sys.argv[1])
	main(sys.argv[1], ["EVE"])

