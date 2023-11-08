import pandas as pd
import os, sys
import json
from scipy.stats import spearmanr

def main(prefix_dir):
	DMS_dir = "~/data/DMS/DMS_tables_GMM/"
	function_dir = "~/data/variant/functional/scores/"
	score_names = ["ESM-2", "ESM-1b", "AlphaMissense", "gMVP", "EVE", "CADD_phred", "REVEL", "PrimateAI", "MPC"]
	summary = pd.read_table(os.path.expanduser(f"{DMS_dir}/summary.txt"))
	for _, row in summary.iterrows():
		uniprot_id = row['UniprotID']
		transcript_id = row['TranscriptID']
		model = os.path.expanduser(f"{prefix_dir}/combined_scores/{uniprot_id}.txt.gz")
		if not os.path.exists(model):
			continue
		model_df = pd.read_table(model)
		dms_filename = os.path.expanduser(f"{DMS_dir}/{row['DMS']}.txt")
		DMS_df = pd.read_table(dms_filename)
		DMS_df = pd.merge(DMS_df, model_df)
		function_df = pd.read_table(os.path.expanduser(f"{function_dir}/{uniprot_id}_scores.txt.gz"))
		DMS_df.rename(columns = {'Protein_position': 'Uniprot_position'})
		function_df = function_df[['Protein_position', 'Uniprot_position', 'AA_ref', 'AA_alt'] + score_names]
#		if function_df['EVE'].count() == 0:
#			EVE_index = score_names.index('EVE')
#			function_df = function_df.dropna(subset = score_names[:EVE_index] + score_names[(EVE_index + 1):])
#		else:
#			function_df = function_df.dropna(subset = score_names)
		function_df = function_df.drop_duplicates(subset = ['Uniprot_position', 'AA_ref', 'AA_alt'])
		DMS_df = pd.merge(function_df, DMS_df)
		if not os.path.exists(os.path.expanduser(f"{prefix_dir}/DMS/")):
			os.makedirs(os.path.expanduser(f"{prefix_dir}/DMS/"))
		DMS_df.to_csv(os.path.expanduser(f"{prefix_dir}/DMS/{row['DMS']}_merged.txt.gz"), sep = "\t", index = False)


if __name__=="__main__":
	main(sys.argv[1])
