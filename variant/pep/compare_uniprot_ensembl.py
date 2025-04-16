from Bio import SeqIO
import gzip
import pandas as pd

def main():
	geneset = pd.read_table("geneset_uniprot.txt")
	
	uniprot_dict = {}
	with gzip.open("uniprot_proteome.fasta.gz", "rt") as f:
		for record in SeqIO.parse(f, "fasta"):
			uniprot_id = record.id.split("|")[1]
			if uniprot_id in geneset['UniprotID'].values:
				uniprot_acc = record.id.split("|")[2].split()[0]
				uniprot_seq = record.seq
				uniprot_dict[uniprot_id] = (uniprot_acc, uniprot_seq)
	
	with gzip.open("gencode.v38.pc_translations.fa.gz", "rt") as f:
		for record in SeqIO.parse(f, "fasta"):
			gencode_id = record.id.split("|")[1].split(".")[0]
			mapped = geneset.loc[geneset['TranscriptID']==gencode_id]
			if len(mapped)!=1:
				continue
			gencode_seq = record.seq
			uniprot_id = mapped['UniprotID'].values[0]
			uniprot_acc, uniprot_seq = uniprot_dict[uniprot_id]
			with gzip.open("uniprot_seq/" + uniprot_id + ".fasta.gz", "wt") as output:
				print(">", uniprot_id, "\n", uniprot_seq, "\n", sep = "", end = "", file = output)
			if gencode_seq != uniprot_seq:
				with gzip.open("unmapped/" + uniprot_id + ".fasta.gz", "wt") as output:
					print(">", uniprot_id, "\n", uniprot_seq, "\n", sep = "", end = "", file = output)
					print(">", gencode_id, "\n", gencode_seq, "\n", sep = "", end = "", file = output)

if __name__=="__main__":
	main()
