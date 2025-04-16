import pandas as pd
import numpy as np
from os.path import exists

def main():
	geneset = pd.read_table("../list/geneset_uniprot_len.txt")
	for _, row in geneset.iterrows():
		uniprot_id = row['UniprotID']
		length = row['Length']
		frac = 1
		filename = f"repr_1000_200/{uniprot_id}_{frac}.npy"
		while exists(filename):
			embedding = np.load(filename)
			for subfrac in range(2):
				new_frac = frac * 2 + subfrac - 1
				prev_end = (new_frac - 2) * 400 + 600
				if (prev_end >= length) & (new_frac > 1):
					break
				np.save(f"repr_600_200/{uniprot_id}_{new_frac}.npy", embedding[(subfrac * 400):(subfrac * 400 + 602)])
			frac += 1
			filename = f"repr_1000_200/{uniprot_id}_{frac}.npy"

if __name__=="__main__":
	main()
