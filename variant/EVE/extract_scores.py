import gzip
import pandas as pd
import os

def main():
	genes_file = open("geneset_uniprot_len.txt", "r")
	output_file = open("EVE_ids.txt", "w")
	genes_file.readline()
	for line in genes_file:
		line_split = line.strip().split("\t")
		id1 = line_split[0]
		id2 = line_split[7]
		filename = "EVE_files/" + id2 + ".csv"
		if (os.path.exists(filename)):
			df = pd.read_csv(filename, low_memory = False)
			df = df[["wt_aa", "position", "mt_aa", "evolutionary_index_ASM", "EVE_scores_ASM", "EVE_classes_75_pct_retained_ASM"]]
			df = df.dropna()
			df.to_csv("EVE_scores/" + id1 + ".txt.gz", sep = "\t", float_format = "%.4f", index = False)
			print(id1, id2, sep = "\t", file = output_file, flush = True)
	genes_file.close()
	output_file.close()

if __name__=="__main__":
	main()
