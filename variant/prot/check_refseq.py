import pandas as pd
from Bio import SeqIO
import gzip

SLIDING = 200

def main():
	geneset = pd.read_table("geneset_uniprot_parts.txt.gz")
	unmapped_file = open("seq_inconsistent.txt", "w")
	for _, row in geneset.iterrows():
		uniprot_id = row['UniprotID']
		parts = row['Parts']
		with gzip.open("../pep/uniprot_seq/"+uniprot_id+".fasta.gz", "rt") as f:
			record = SeqIO.read(f, "fasta")
		complete_seq = record.seq
		for frac in range(parts):
			AF_table = pd.read_table(f"AF2_table/coords_{uniprot_id}-F{frac+1}.gz")
			length = len(AF_table)
			partial_seq = AF_table['refAA'].tolist()
			partial_seq = "".join(partial_seq)
			start = frac * SLIDING
			if complete_seq[start:(start+length)]!=partial_seq:
				print(uniprot_id, file = unmapped_file)
				break
	unmapped_file.close()
		


if __name__=="__main__":
	main()
