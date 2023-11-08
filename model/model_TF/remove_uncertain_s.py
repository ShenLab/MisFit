import pandas as pd
from constants import log_dir, setting
import os

def main():
	covered = []
	with open("combine_covered_list.txt", "r") as f:
		for line in f:
			covered.append(line.strip())
	geneset = pd.read_table(os.path.expanduser(setting['geneset']))
	for _, row in geneset.iterrows():
		uniprot_id = row['UniprotID']
		if uniprot_id not in covered:
			filename = f"{log_dir}/combined_scores/{uniprot_id}.txt.gz"
			if os.path.exists(filename):
				df = pd.read_table(filename)
				df['model_selection'] = pd.NA
				df.to_csv(filename, index = False, sep = "\t")


if __name__ == "__main__":
	main()
