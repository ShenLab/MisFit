import numpy as np
import pandas as pd

def main():
	geneset = pd.read_table("../list/geneset_uniprot_len.txt")
	for _, row in geneset.iterrows():
		uniprot_id = row['UniprotID']
		count = np.load(f"MSA_count/{uniprot_id}_count.npy")
		count_sum = np.sum(count, axis = -1, keepdims = True)
		count[count != 0] = 1
#		new_count = count_sum / np.sum(count, axis = -1, keepdims = True) * count
#		np.save(f"MSA_one/{uniprot_id}_count.npy", new_count)
		np.save(f"MSA_one/{uniprot_id}_count.npy", count)
if __name__=="__main__":
	main()
