from Bio import SeqIO
import gzip
import pandas as pd
import re

def main():
	geneset = pd.read_table("geneset.txt")
	
	with gzip.open("Homo_sapiens.GRCh38.pep.all.fa.gz", "rt") as f:
		for record in SeqIO.parse(f, "fasta"):
			protein_id = record.id.split(".")[0]
			mapped = geneset.loc[geneset['ProteinID']==protein_id]
			if len(mapped)!=1:
				continue
			gencode_id = mapped['TranscriptID'].values[0]
			gencode_seq = record.seq
			with gzip.open("ensembl_seq/" + gencode_id + ".fasta.gz", "wt") as output:
				print(">", gencode_id, "\n", gencode_seq, "\n", sep = "", end = "", file = output)

if __name__=="__main__":
	main()
