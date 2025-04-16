import numpy as np
import pandas as pd
import tensorflow as tf

def main():
	geneset = pd.read_table("../list/geneset_uniprot.txt")
	MSA_folder = "MSA_orthologues/"
	sum_same_aa = 0.
	sum_l = 0.
	for _, row in geneset.iterrows():
		uniprot_id = row['UniprotID']
		MSA = np.load(f"{MSA_folder}/{uniprot_id}_MSA.npy")
		# L, N, A
		MSA = tf.one_hot(MSA, depth = 21)
		same_aa = tf.einsum('lna,lma->nml', MSA, MSA)
		sum_same_aa += tf.reduce_sum(same_aa, axis = -1)
		sum_l += same_aa.shape[-1]
	similarity_matrix = sum_same_aa / sum_l
	np.save("similarity_matrix.npy", similarity_matrix.numpy())
	similarity_weight = 1 / tf.reduce_sum(similarity_matrix ** 2, axis = -1)
	np.save("similarity_weight.npy", similarity_weight.numpy())

if __name__=="__main__":
	main()
