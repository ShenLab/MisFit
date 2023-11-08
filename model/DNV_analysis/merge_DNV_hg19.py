import pandas as pd
import os, sys
import json
from scipy.stats import spearmanr

def main(prefix_dir, condition = "ASD"):
	ESM_dir = "~/data/variant/ESM-2/logits_merged/"
	function_dir = "~/data/variant/functional/scores/"
	score_names = ["AlphaMissense", "ESM-1b", "ESM-2", "gMVP", "EVE", "CADD_phred", "REVEL", "PrimateAI", "MPC"]
	geneset = pd.read_table(os.path.expanduser(f"~/data/variant/list/geneset_uniprot_len.txt"))
	DNV_df = pd.read_table(os.path.expanduser(f"~/data/model/DNV_analysis/{condition}_DNV_mis_hg19.txt.gz"))
	DNV_df2 = pd.DataFrame()
	for _, row in geneset.iterrows():
		uniprot_id = row['UniprotID']
		transcript_id = row['TranscriptID']
		DNV_df_sub = DNV_df[DNV_df['TranscriptID']==transcript_id]
		if len(DNV_df_sub) == 0:
			continue
		model = os.path.expanduser(f"{prefix_dir}/combined_scores/{uniprot_id}.txt.gz")
		if not os.path.exists(model):
			continue
		model_df = pd.read_table(model)
		model_df = model_df.rename(columns = {'Protein_position': 'Uniprot_position'})
		function_df = pd.read_table(os.path.expanduser(f"{function_dir}/{uniprot_id}_scores.txt.gz"))
		function_df = function_df[['Protein_position', 'AA_ref', 'AA_alt', 'Uniprot_position'] + score_names].drop_duplicates(subset = ['Protein_position', 'AA_ref', 'AA_alt'])
		DNV_df_sub = pd.merge(DNV_df_sub, function_df, how = "left")
		model_df = pd.merge(DNV_df_sub, model_df, how = "left")
		DNV_df2 = pd.concat([DNV_df2, model_df])

	if not os.path.exists(os.path.expanduser(f"{prefix_dir}/DNV/")):
		os.makedirs(os.path.expanduser(f"{prefix_dir}/DNV/"))
	
	DNV_df2 = pd.merge(DNV_df2, geneset[['UniprotID', 'TranscriptID', 'GeneID', 'Symbol']])
	DNV_df2.to_csv(os.path.expanduser(f"{prefix_dir}/DNV/{condition}_mis_hg19_merged.txt.gz"), sep = "\t", index = False)


if __name__=="__main__":
	main(sys.argv[1], "ASD")
	main(sys.argv[1], "NDD")
#	main(sys.argv[1], "EA")
