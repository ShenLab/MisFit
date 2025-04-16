import numpy as np
import pandas as pd

def main():
	geneset = pd.read_table("../list/geneset_uniprot.txt")
	all_cover = 0
	for _, row in geneset.iterrows():
		uniprot_id = row['UniprotID']
		cover = np.load(f"MSA_orthologues/{uniprot_id}_cover.npy")
		all_cover += cover
	f = open("ensembl_species_cover.txt", "w")
	for sp in all_cover.tolist()[1:]:
		print(int(sp), file = f)
	f.close()

if __name__ == "__main__":
	main()
