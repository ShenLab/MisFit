import numpy as np
import pandas as pd
import tensorflow as tf

def main():
	geneset = pd.read_table("../list/geneset_uniprot_len.txt")
	MSA_folder = "MSA_orthologues/"
	sum_same_aa = 0.1
	sum_l = 0.1
	n = 0
	geneset_sample = geneset[(geneset['Length'] > 200) & (geneset['Length'] < 1000)].sample(n = 2000)
	for _, row in geneset_sample.iterrows():
		uniprot_id = row['UniprotID']
		MSA = np.load(f"{MSA_folder}/{uniprot_id}_MSA.npy")
		# L, N, A
		MSA = tf.one_hot(MSA, depth = 20)
		same_aa = tf.einsum('lna,lma->nml', MSA, MSA)
		not_gap = tf.reduce_sum(MSA, axis = -1)
		pair_not_gap = tf.einsum('ln,lm->nml', not_gap, not_gap)
		sum_same_aa += tf.reduce_sum(same_aa, axis = -1)
		sum_l += tf.reduce_sum(pair_not_gap, axis = -1)
		n += 1
		if n % 100 == 0:
			print(f"{n} proteins processed.")
	similarity_matrix = sum_same_aa / sum_l
	np.save("similarity_matrix.npy", similarity_matrix.numpy())
	similarity_weight = tf.reduce_sum(1 - similarity_matrix, axis = -1) + 1
	similarity_weight = similarity_weight / similarity_weight.shape[0]
	np.save("similarity_weight.npy", similarity_weight.numpy())

if __name__=="__main__":
	main()
