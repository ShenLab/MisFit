import json
from os.path import exists
import random
import pandas as pd

ratio = 0.9
random.seed(0)

def main():
	df = pd.read_table("PrimateAI_benign.txt.gz")
	for _, row in df.iterrows():
		uniprot_id = row['UniprotID']
		AA_ref = row['AA_ref']
		AA_alt = row['AA_alt']
		AA_pos = row['Protein_position']
		target = 0
		if random.random() < ratio:
			split = 1
		else:
			split = 0
		with open("var_by_uniprot/" + uniprot_id + ".txt", "a") as outfile:
			print(uniprot_id, AA_ref, AA_pos, AA_alt, target, split, sep = "\t", file = outfile)

			

if __name__=="__main__":
	main()
