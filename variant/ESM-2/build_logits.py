import numpy as np
import json
import pandas as pd

A = 20

# files used are already in order of position and altAA
def main():
	geneset = pd.read_table("geneset_uniprot_len.txt")
	for _, row in geneset.iterrows():
		uniprot_id = row['UniprotID']
		data = pd.read_table(f"logits_merged/{uniprot_id}.txt.gz")
		logits = data['ESM'].to_numpy(dtype = np.float32)
		logits = np.reshape(logits, (-1, A))
		np.save(f"logits_np/{uniprot_id}.npy", logits)

if __name__=="__main__":
	main()



