import pandas as pd

def main():
	# list_1 for training initial NN (constrained, autochrom, not too short)
	# list_2 for learning gene parameters (all chromosomes with AN available)
	# list_3 for testing on individual genes (DMS)
	min_misz = 2.
	min_pLI = 0.5
	min_length = 200
	test_list = pd.read_table("../../../DMS/DMS_tables/summary.txt")

	all_seg = pd.read_table("../seg_list_600_200.txt")
	constraint = pd.read_table("../gnomad.v2.1.1.lof_metrics.by_gene.txt.gz")
	covered = set()
	with open("../../snv_info_2/combine_covered_list.txt", "r") as f:
		for line in f:
			covered.add(line.strip())
	exclude = set()
	depth = pd.read_table("../../zoonomia/orthologues_depth.txt")
	
#	with open("../../prot/seq_inconsistent.txt", "r") as f:
#		for line in f:
#			exclude.add(line.strip())
	# 0: all with structure; 1: constrained in 2; 2: all covered; 3. test gene only
	list_0 = all_seg[~all_seg['UniprotID'].isin(exclude)]
	list_2 = list_0[list_0['UniprotID'].isin(covered)]
	list_3 = list_0[list_0['UniprotID'].isin(test_list['UniprotID'])]

	constraint_list = constraint[(constraint["pLI"]>=min_pLI) | (constraint["mis_z"]>=min_misz)]["gene_id"].tolist()
	depth_list = depth[depth['depth'] > 350]['UniprotID'].tolist()
	list_2 = list_2[list_2['UniprotID'].isin(depth_list)]
	list_2 = list_2[list_2['end'] - list_2['start'] + 1 > min_length]
	list_1 = list_2[list_2['GeneID'].isin(constraint_list)]
	list_1 = list_1[list_1['Chrom'].isin([str(i) for i in range(1, 23)])]
#	list_1 = list_1[list_1['UniprotID'].isin(depth_list)]
	list_0.to_csv("list_0.txt", sep = "\t", index = False)
	list_1.to_csv("list_1.txt", sep = "\t", index = False)
	list_2.to_csv("list_2.txt", sep = "\t", index = False)
	list_3.to_csv("list_3.txt", sep = "\t", index = False)

	with open("summary.txt", "w") as f:
		print(0, len(list_0['UniprotID'].unique()), len(list_0), sep = "\t", file = f)
		print(1, len(list_1['UniprotID'].unique()), len(list_1), sep = "\t", file = f)
		print(2, len(list_2['UniprotID'].unique()), len(list_2), sep = "\t", file = f)
		print(3, len(list_3['UniprotID'].unique()), len(list_3), sep = "\t", file = f)

if __name__=="__main__":
	main()
