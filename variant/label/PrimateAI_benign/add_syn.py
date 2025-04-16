from Bio import SeqIO
import pandas as pd
import random
import gzip

ratio = 0.9
random.seed(0)

def main():	
	geneset = pd.read_table("../../list/geneset_uniprot.txt")
	for _, row in geneset.iterrows():
		uniprot_id = row['UniprotID']
		with gzip.open(f"../../pep/uniprot_seq/{uniprot_id}.fasta.gz", "rt") as f:
			record = SeqIO.read(f, "fasta")
		seq = record.seq
		with open(f"var_syn_by_uniprot/{uniprot_id}.txt", "a") as f:
			for i in range(len(seq)):
				AA_ref = seq[i]
				AA_alt = seq[i]
				target = 0
				if random.random() < ratio:
					split = 1
				else:
					split = 0
				print(uniprot_id, AA_ref, i + 1, AA_alt, target, split, sep = "\t", file = f)

if __name__=="__main__":
	main()

