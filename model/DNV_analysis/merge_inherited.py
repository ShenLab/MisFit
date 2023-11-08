import pandas as pd
import os, sys
import json
from scipy.stats import spearmanr

def main(prefix_dir):
#	ESM_dir = "~/data/variant/ESM-2/logits_merged/"
#	function_dir = "~/data/variant/functional/scores/"
#	score_names = ["gMVP", "EVE", "CADD_phred", "REVEL", "PrimateAI", "MPC"]
	function_dir = "~/data/variant/functional/snv_only/"
	pop_dir = "~/data/variant/snv_info_2/mis_info_by_protein_mapping/"
	pop_names = ['UKBB', 'gnomAD_NFE_genome', 'gnomAD_NFE_exome']
	AC_names = [pop_name + "_AC" for pop_name in pop_names]
	AN_names = [pop_name + "_AN" for pop_name in pop_names]
	geneset = pd.read_table(os.path.expanduser(f"~/data/variant/list/geneset_uniprot_len.txt"))
	DNV_df = pd.read_table(os.path.expanduser(f"~/data/model/DNV_analysis/ASD_SPARK0_missense.txt.gz"))
	DNV_df = DNV_df.rename(columns = {'Protein_position': 'Ensembl_protein_position'})
	DNV_df['Chrom'] = DNV_df['Chrom'].astype('str')
	DNV_df2 = pd.DataFrame()
	for _, row in geneset.iterrows():
		uniprot_id = row['UniprotID']
		gene_id = row['GeneID']
		DNV_df_sub = DNV_df[DNV_df['GeneID']==gene_id]
		if len(DNV_df_sub) == 0:
			continue
		model = os.path.expanduser(f"{prefix_dir}/combined_scores/{uniprot_id}.txt.gz")
		if not os.path.exists(model):
			continue
		model_df = pd.read_table(model)
#		ESM_filename = os.path.expanduser(f"{ESM_dir}/{uniprot_id}.txt.gz")
#		ESM_df = pd.read_table(ESM_filename)
#		model_df = pd.merge(ESM_df, model_df)
		model_df = model_df.rename(columns = {'Protein_position': 'Uniprot_position'})
		function_df = pd.read_table(os.path.expanduser(f"{function_dir}/{uniprot_id}.txt.gz"))
#		function_df = pd.read_table(os.path.expanduser(f"{function_dir}/{uniprot_id}_scores.txt.gz"))
#		function_df = function_df[['Chrom', 'Pos', 'Ref', 'Alt', 'AA_ref', 'AA_alt', 'Uniprot_position'] + score_names]
		function_df['Chrom'] = function_df['Chrom'].astype('str')
		DNV_df_sub = pd.merge(DNV_df_sub, function_df)
		model_df = pd.merge(DNV_df_sub, model_df)
		pop_filename = os.path.expanduser(f"{pop_dir}/{uniprot_id}_info.txt.gz")
		if os.path.exists(pop_filename):
			pop_df = pd.read_table(pop_filename)
			pop_df['AC'] = pop_df[AC_names].sum(axis = 1)
			pop_df['AN'] = pop_df[AN_names].sum(axis = 1)
			model_df = pd.merge(model_df, pop_df[['Pos', 'Ref', 'Alt', 'AN', 'AC', 'outlier', 'Filter']], how = "left")
		else:
			model_df['AC'] = 0
			model_df['AN'] = 0
		DNV_df2 = pd.concat([DNV_df2, model_df])
	if not os.path.exists(os.path.expanduser(f"{prefix_dir}/DNV/")):
		os.makedirs(os.path.expanduser(f"{prefix_dir}/DNV/"))
	
	DNV_df2.to_csv(os.path.expanduser(f"{prefix_dir}/DNV/ASD_SPARK0_missense_merged.txt.gz"), sep = "\t", index = False)


if __name__=="__main__":
	main(sys.argv[1])
