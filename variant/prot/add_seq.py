import gzip
from Bio import SeqIO

def main():
	struct_max_l = 2700
	sliding_l = 1400
	sliding_window = 200
	seq_dir = "../pep/uniprot_seq/"
	f = open("seq_inconsistent.txt", "r")
	summary = open("add_frac_seq.txt", "w")
	for line in f:
		uniprot_id = line.strip()
		seq_file = gzip.open(f"{seq_dir}/{uniprot_id}.fasta.gz", "rt")
		record = SeqIO.read(seq_file, "fasta")
		seq_file.close()
		length = len(record.seq)
		if length <= struct_max_l:
			with open(f"./add_alphafold2_seq/{uniprot_id}_F1.fasta", "w") as output:
				print(f">{uniprot_id}_F1\n{record.seq}", file = output)
				print(f"{uniprot_id}_F1", file = summary)
		else:
			for i in range(length // sliding_window + 1):
				start = sliding_window * i
				end = start + sliding_l
				with open(f"./add_alphafold2_seq/{uniprot_id}_F{i+1}.fasta", "w") as output:
					print(f">{uniprot_id}_F{i+1}\n{record.seq[start:end]}", file = output)
					print(f"{uniprot_id}_F{i+1}", file = summary)
				if end >= length:
					break
	f.close()
	summary.close()


if __name__=="__main__":
	main()
