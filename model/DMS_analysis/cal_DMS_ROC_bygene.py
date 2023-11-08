import pandas as pd
import numpy as np
import os, sys
import json
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve, matthews_corrcoef

def mcc(tp, tn, fp, fn):
	sup = tp * tn - fp * fn
	inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
	if inf==0:
		return 0
	else:
		return sup / np.sqrt(inf)

def eval_mcc(y_true, y_prob):
	idx = np.argsort(y_prob)
	y_true_sort = y_true[idx]
	n = y_true.shape[0]
	nump = 1.0 * np.sum(y_true) # number of positive
	numn = n - nump # number of negative
	tp = nump
	tn = 0.0
	fp = numn
	fn = 0.0
	best_mcc = 0.0
	best_id = -1
	prev_proba = -1
	best_proba = -1
	mccs = np.zeros(n)
	for i in range(n):
		proba = y_prob[idx[i]]
		if proba != prev_proba:
			prev_proba = proba
			new_mcc = mcc(tp, tn, fp, fn)
			if new_mcc >= best_mcc:
				best_mcc = new_mcc
				best_id = i
				best_proba = proba
		if y_true_sort[i] == 1:
			tp -= 1.0
			fn += 1.0
		else:
			fp -= 1.0
			tn += 1.0
	return best_proba

def main(prefix_dir):
	exclude_list = ['DDX3X', 'BRCA2']
	DMS_dir = "~/data/DMS/DMS_tables_GMM/"
	score_names = ["model_damage", "model_selection", "ESM-1b", "ESM-2", "AlphaMissense", "gMVP", "EVE", "CADD_phred", "REVEL", "PrimateAI", "MPC"]
	result = {'UniprotID': [], 'num_AA_change': []}
	threshold = {'UniprotID': []}

	for score_name in score_names:
		result[score_name] = []
		threshold[score_name] = []
	summary = pd.read_table(os.path.expanduser(f"{DMS_dir}/summary.txt"))
	summary = summary[~summary['Symbol'].isin(exclude_list)]
	summary = summary[(summary['exist_label'] == 1) | ((summary['benign_proportion'] > 0.4) & (summary['benign_proportion'] + summary['damaging_proportion'] > 0.9))]
	DMS_df_all = pd.DataFrame()
	for _, row in summary.iterrows():
		uniprot_id = row['UniprotID']
		DMS_merged = os.path.expanduser(f"{prefix_dir}/{row['DMS']}_merged.txt.gz")
		if not os.path.exists(DMS_merged):
			continue
		DMS_df = pd.read_table(DMS_merged)
		if row['exist_label'] == 0:
			DMS_df['target'] = DMS_df['GMM_target']
		DMS_df = DMS_df.dropna(subset = ["target"])
		score_names_anal = []
		for score_name in score_names:
			if score_name in DMS_df.columns:
				if DMS_df[score_name].count() > (0.2 * DMS_df.shape[0]):
					score_names_anal.append(score_name)
		DMS_df = DMS_df.dropna(subset = score_names_anal)
		DMS_df['UniprotID'] = uniprot_id
		DMS_df['Symbol'] = row['Symbol']
		DMS_df_all = pd.concat([DMS_df_all, DMS_df[['UniprotID', 'Symbol', 'Protein_position', 'AA_ref', 'AA_alt', 'target'] + score_names]])

	DMS_df_all = DMS_df_all.sort_values(by = 'target', ascending = False)
	DMS_df_all = DMS_df_all.drop_duplicates(subset = ['Symbol', 'Protein_position', 'AA_alt'])
	DMS_df_all = DMS_df_all.sort_values(by = ['Symbol', 'Protein_position', 'AA_alt']).reset_index(drop = True)
	DMS_df_all.to_csv(os.path.expanduser(f"{prefix_dir}/DMS_combined_label.txt.gz"), index = False, sep = "\t")
	
	for score_name in score_names:
		DMS_df_all[f"{score_name}_rank"] = DMS_df_all[score_name].rank(pct = True)
	
	for uniprot_id in DMS_df_all['UniprotID'].unique():
		result['UniprotID'].append(uniprot_id)
		threshold['UniprotID'].append(uniprot_id)
		DMS_df = DMS_df_all[DMS_df_all['UniprotID'] == uniprot_id]
		result['num_AA_change'].append(len(DMS_df))
		for score_name in score_names:
			if DMS_df[score_name].count() > (0.2 * DMS_df.shape[0]):
				DMS_df_clean = DMS_df[['target', score_name]].dropna()
				AU = roc_auc_score(DMS_df_clean['target'], DMS_df_clean[score_name])
				best_thresh = eval_mcc(DMS_df_clean['target'].to_numpy(), DMS_df_clean[score_name].to_numpy())
			else:
				AU = np.nan
				best_thresh = np.nan
			result[score_name].append(AU)
			threshold[score_name].append(best_thresh)

	DMS_df = DMS_df_all
	threshold['UniprotID'].append('all')
	for score_name in score_names:
		DMS_df_clean = DMS_df[['target', score_name]].dropna()
		best_thresh = eval_mcc(DMS_df_clean['target'].to_numpy(), DMS_df_clean[score_name].to_numpy())
		threshold[score_name].append(best_thresh)
	result_df = pd.DataFrame(result)
	result_df = pd.merge(result_df, summary[['UniprotID', 'Symbol']].drop_duplicates(), how = "left")
	result_df.to_csv(os.path.expanduser(f"{prefix_dir}/AUC_combined_summary.txt"), sep = "\t", index = False)
	thresh_df = pd.DataFrame(threshold)
	thresh_df = pd.merge(thresh_df, summary[['UniprotID', 'Symbol']].drop_duplicates(), how = "left")
	thresh_df.to_csv(os.path.expanduser(f"{prefix_dir}/thresh_combined_summary.txt"), sep = "\t", index = False)

	for i, row in thresh_df.iterrows():
		for score_name in score_names:
			if pd.notna(thresh_df.at[i, score_name]):
				idx = (DMS_df_all[score_name] - row[score_name]).abs().idxmin()
				thresh_df.at[i, score_name]=DMS_df_all.at[idx,f"{score_name}_rank"]
	
	thresh_df.to_csv(os.path.expanduser(f"{prefix_dir}/thresh_rank_combined_summary.txt"), sep = "\t", index = False)


if __name__=="__main__":
	main(sys.argv[1])
