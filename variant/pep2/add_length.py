from Bio import SeqIO
import gzip
import pandas as pd
import re

def main():
	geneset = pd.read_table("geneset.txt")
	geneset['Length'] = pd.NA
	for i, row in geneset.iterrows():
		transcript_id = row['TranscriptID']
		with gzip.open("ensembl_seq/" + transcript_id + "fasta.gz", "rt") as f:
			record = SeqIO.read(f, "fasta")
		geneset.at[i, 'Length'] = len(record.seq)
	geneset.to_csv("geneset_len.txt", sep = "\t", index = False)

if __name__=="__main__":
	main()
