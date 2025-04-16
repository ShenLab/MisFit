import tensorflow as tf
import json
from Bio import SeqIO
import gzip
import pandas as pd
import numpy as np
from os.path import exists

MAX_L = 1000
SLIDING = 800
OVERLAP = MAX_L - SLIDING

with open("setting.json", "r") as f:
	settings = json.load(f)
A = len(settings['AA_table']) + 1 # Amino Acid dictionary size + gap

def get_cov(msa_feature, preserve_all = False):
	# msa_feature: tensor with shape (L, N) sequence length * number of aligned sequences, each element is an integer amino acid index within range (0, A)
	L = msa_feature.shape[0]
	N = msa_feature.shape[1]
	assert L <= MAX_L
	if L < MAX_L:
		msa_feature = tf.concat([msa_feature, tf.ones((MAX_L - L, N), dtype = tf.int32) * (A - 1)], axis = 0)
		L = MAX_L
	
	W = tf.constant(1.0 / N, shape=(1, N), dtype=tf.float32)
	x = tf.one_hot(tf.cast(msa_feature, tf.int32), depth=A, axis=-1) # (L, N, A)

	x1 = tf.matmul(W[:, tf.newaxis], x) # (L, 1, A)
	x2 = x - x1 # (L, N, A)
	x2 = tf.sqrt(W[:, :, tf.newaxis]) * x2 # (L, N, A)

	x2_t = tf.reshape(tf.transpose(x2, perm=(1, 0, 2)), shape=(N, L * A))

	x3 = tf.matmul(tf.transpose(x2, perm=(0, 2, 1)), x2_t) # (L, A, L * A)
	x3 = tf.reshape(x3, (L, A, L, A))
	x3 = tf.transpose(x3, (0, 2, 1, 3)) # (L, L, A, A) cov of per pair pos, per pair AA type
	x3 = tf.reshape(x3, (L, L, A * A))
	norm = tf.sqrt(tf.reduce_sum(tf.square(x3), axis=-1) + 1e-12) # result: (L, L) cov of pair pos
	if preserve_all:
		x4 = tf.concat([x3, tf.expand_dims(norm, -1)], axis = -1) # result: (L, L, A * A + 1)
		return x4
	else:
		return norm

def seq_to_list(seq):
	return [settings['AA_table'].get(AA, A-1) for AA in seq]

def aligned_to_tensor(fasta_filename):
	seq_list = []
	with gzip.open(fasta_filename, "rt") as f:
		for record in SeqIO.parse(f, "fasta"):
			seq_list.append(seq_to_list(record.seq))
	return  tf.transpose(tf.constant(seq_list, dtype = tf.int32))

def main():
	MSA_dir = "MSA_by_uniprot"
	cov_dir = "coev_norm_" + str(MAX_L) + "_" + str(OVERLAP)
	
	df = pd.read_table("../list/geneset_uniprot_len.txt")

	for index, row in df.iterrows():
		uniprot_id = row['UniprotID']
		length = row['Length']
		fasta_filename = f'{MSA_dir}/{uniprot_id}_aligned.fasta.gz'
		tensor = aligned_to_tensor(fasta_filename)
		if tensor.shape[0] != length:
			print(f"{uniprot_id} length not match: MSA {tensor.shape[0]}, Uniprot {length}")
			continue

		for part in range(length // SLIDING + 1):
			start = SLIDING * part
			end = start + MAX_L
			filename = f'{cov_dir}/{uniprot_id}_{part + 1}.npy'
#			if exists(filename):
#				continue
			if end >= length:
				end = length
				coev_norm = get_cov(tensor[start:end])
				np.save(filename, coev_norm.numpy())
				break
			else:
				coev_norm = get_cov(tensor[start:end])
				np.save(filename, coev_norm.numpy())

	if index % 1000 == 0:
		print(f"{index} genes processed.")

if __name__=="__main__":
	main()

