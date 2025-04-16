import pandas as pd

def main():
	geneset_dt = pd.read_table("../list/geneset.txt")
	peplength_dt = pd.read_table("../pep/all_protein_length.txt", names = ['ProteinID', 'peplength'])
	lof_dt = pd.read_table("sum_lof_gene.txt", names = ['GeneID', 'lof_count', 'lof_sumAN', 'lof_AC', 'lof_mu'])
	lof_dt['lof_AN'] = lof_dt['lof_sumAN'] / lof_dt['lof_count']
	syn_dt = pd.read_table("sum_syn_gene.txt", names = ['GeneID', 'syn_count', 'syn_sumAN', 'syn_AC', 'syn_mu'])
	syn_dt['syn_AN'] = syn_dt['syn_sumAN'] / syn_dt['syn_count']
	geneset_merged_dt = geneset_dt.merge(peplength_dt.drop_duplicates(), how = "left", on = "ProteinID")
	geneset_merged_dt = geneset_merged_dt.merge(
		lof_dt[['GeneID', 'lof_count', 'lof_AC', 'lof_AN', 'lof_mu']], on = "GeneID", how = "left"
	).merge(
		syn_dt[['GeneID', 'syn_count', 'syn_AC', 'syn_AN', 'syn_mu']], on = "GeneID", how = "left"
	)
	geneset_merged_dt.to_csv("geneset_summary.csv", index = False)

if __name__ == "__main__":
	main()
