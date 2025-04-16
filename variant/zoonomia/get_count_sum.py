import numpy as np
import pandas as pd

def main():
	geneset = pd.read_table("../list/geneset_uniprot_len.txt")
	n = 0
	for _, row in geneset.iterrows():
		uniprot_id = row['UniprotID']
		count = np.load(f"MSA_one/{uniprot_id}_count.npy")
		n += np.sum(count) - count.shape[0]
	print(n)
if __name__=="__main__":
	main()
