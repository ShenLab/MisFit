import numpy as np
import pandas as pd
import tensorflow as tf

def main():
	geneset = pd.read_table("../list/geneset_uniprot.txt")
	MSA_folder = "MSA_orthologues/"
	similarity_weight = np.load("similarity_weight.npy") # (N,)
	for _, row in geneset.iterrows():
		uniprot_id = row['UniprotID']
		MSA = np.load(f"{MSA_folder}/{uniprot_id}_MSA.npy")
		# L, N, A
		MSA = tf.one_hot(MSA, depth = 20)
		MSA_count = tf.reduce_sum(MSA * tf.expand_dims(similarity_weight, -1), axis = 1) # (L, A)
		np.save(f"MSA_count/{uniprot_id}_count.npy", MSA_count.numpy())

if __name__=="__main__":
	main()
