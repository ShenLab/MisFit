import pandas as pd
import numpy as np
from os.path import exists
from scipy.stats import beta
import sys

def main():
	geneset = pd.read_table("../list/geneset_uniprot_len.txt")
	summary_list = set()
	discrepancy_rate = 50
	q = 0.005
	prior = 0.1
	dirs = ['mis_info_by_protein_mapping', 'syn_info_by_protein_mapping', 'lof_info_by_protein']	
	popnames = ['UKBB', 'gnomAD_NFE_genome', 'gnomAD_NFE_exome']
	for _, row in geneset.iterrows():
		uniprot_id = row['UniprotID']
		for folder in dirs:
			filename = f"{folder}/{uniprot_id}_info.txt.gz"
			if exists(filename):
				df = pd.read_table(filename)
				if len(df[df['outlier'] == True]) == 0:
					continue
				print(uniprot_id)
				summary_list.add(uniprot_id)
				min_upper_bound = 1.
				max_lower_bound = 0.
				for popname in popnames:
					curr_upper_bound = beta.ppf(1 - q, df[f'{popname}_AC'] + prior, df[f'{popname}_AN'] - df[f'{popname}_AC'] + prior)
					min_upper_bound = np.minimum(min_upper_bound, curr_upper_bound)
					curr_lower_bound = beta.ppf(q, df[f'{popname}_AC'] + prior, df[f'{popname}_AN'] - df[f'{popname}_AC'] + prior)
					max_lower_bound = np.maximum(max_lower_bound, curr_lower_bound)
				need_mask = discrepancy_rate * min_upper_bound < max_lower_bound
				df['outlier'] = need_mask
				df.to_csv(filename, index = False, sep = "\t")
	f = open("discr_list.txt", "w")
	for x in summary_list:
		print(x, file = f)
	f.close()

if __name__=="__main__":
	main()
