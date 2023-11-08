import pandas as pd
from constants import log_dir, setting
import os

def main():
	covered = []
	with open("combine_covered_list.txt", "r") as f:
		for line in f:
			covered.append(line.strip())
	geneset = pd.read_table(f"{log_dir}/geneset_mis_s.txt")
	geneset.loc[~geneset['UniprotID'].isin(covered), 'logit_s_mean'] = pd.NA
	geneset.loc[~geneset['UniprotID'].isin(covered), 'logit_s_sd'] = pd.NA
	geneset.to_csv(f"{log_dir}/geneset_mis_s.txt", index = False, sep = "\t")


if __name__ == "__main__":
	main()
