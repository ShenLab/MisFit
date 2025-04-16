import numpy as np
import tensorflow as tf
import pandas as pd

def main():
	MSA = np.load("MSA_genetree/P60484_MSA.npy")
	MSA_one_hot = tf.one_hot(MSA, 21)
	AA_count = tf.reduce_sum(MSA_one_hot, axis = 1)
	AA_count += 0.01
	p = AA_count / tf.reduce_sum(AA_count, axis = -1, keepdims = True)
	entropy = -tf.reduce_sum(tf.math.log(p) * p, axis = -1)
	df = pd.DataFrame({"entropy": entropy, "Protein_position": [i+1 for i in range(len(entropy))]})
	df.to_csv("P60484_conservation.txt", sep = "\t", index = False)

if __name__ == "__main__":
	main()
