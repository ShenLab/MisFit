import pandas as pd
import numpy as np

def main():
	df = pd.read_table("result/geneset_lof_s.txt")
	df2 = pd.read_table("s_het_Weghorn.txt")
	df2 = df2[['Gene', 's_het_drift', 'low_drift', 'up_drift']]
	df2.columns = ['Symbol', 's_mean_Weghorn', 's_lower_Weghorn', 's_upper_Weghorn']
	df = pd.merge(df, df2, how = 'left')
	df3 = pd.read_table("s_het_Agarwal.txt")
	df3 = df3[['Gene', 'log10_ci_low', 'log10_ci_high', 'log10_map']]
	df3.columns = ['Symbol', 's_lower_Agarwal', 's_upper_Agarwal', 's_mode_Agarwal']
	for colname in ['s_lower_Agarwal', 's_upper_Agarwal', 's_mode_Agarwal']:
		df3[colname] = 10 ** df3[colname]
	df = pd.merge(df, df3, how = "left")
	df4 = pd.read_table("gnomad.v2.1.1.lof_metrics.by_gene.txt.gz")
	df4 = df4[['gene_id', 'oe_lof', 'oe_mis', 'pLI', 'lof_z', 'mis_z']]
	df4.columns = ['GeneID', 'oe_lof', 'oe_mis', 'pLI', 'lof_z', 'mis_z']
	df = pd.merge(df, df4, how = 'left')
	df5 = pd.read_table("s_het_Zeng.tsv")
	df5 = df5[['ensg', 'post_mean', 'post_lower_95', 'post_upper_95']]
	df5.columns = ['GeneID', 's_mean_Zeng', 's_lower_Zeng', 's_upper_Zeng']
	df = pd.merge(df, df5, how = "left")
	df.to_csv("geneset_lof_s_combined.txt", index = False, sep = "\t")

if __name__=="__main__":
	main()
