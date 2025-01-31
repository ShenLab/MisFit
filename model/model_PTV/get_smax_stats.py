import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.neighbors import KernelDensity

def main():
	result_dir = "result"
	log_s_mean = np.load(f"{result_dir}/smax_mean_gene.npy") # (ngene, 1)
	geneset = pd.read_table("../../variant/list/geneset_uniprot_len.txt")
	geneset[['logit_s_mean']] = log_s_mean

	# gene with no coverage of AC
	exclude = set()
	with open("../../variant/snv_info_2/combine_covered_list_lof.txt", "r") as f:
		for line in f:
			exclude.add(line.strip())
	geneset.loc[(~geneset['UniprotID'].isin(exclude))|(geneset['Chrom']=="Y"), ['logit_s_mean']] = pd.NA
	geneset.to_csv(f"{result_dir}/geneset_lof_s.txt", index = False, sep = "\t")

if __name__=="__main__":
	main()
