from pyfaidx import Fasta
import random
import sys
from time import time
from Bio import SeqIO
import gzip

def create_dict():
	count_dict = {}
	for i in ['A', 'C', 'G', 'T']:
		for j in ['A', 'C', 'G', 'T']:
			for k in ['A', 'C', 'G', 'T']:
				count_dict[i+j+k] = 0
	return count_dict

def main():
#	genome = Fasta("../../annotation/.vep/homo_sapiens/104_GRCh38/Homo_sapiens.GRCh38.dna.toplevel.fa.gz")
	file = gzip.open("Homo_sapiens.GRCh38.dna_rm.primary_assembly.fa.gz","rt")
	genome = SeqIO.to_dict(SeqIO.parse(file, "fasta"))
	file.close()
	count_dict = create_dict()
	count = 0
	valid_count = 0
	f = open("nonexon_region.txt", "r")
	timea = time()
	for line in f:
		count += 1
		if count % 5000 == 0:
			timeb = time()
			print(f"{valid_count} in {count}, {int((timeb-timea)/60)}")
		line_split = line.strip().split("\t")
		chrom = line_split[0]
		start = int(line_split[1])
		end = int(line_split[2])
		start = random.randint(start, end - 301)
		end = start + 300
		seq = str(genome[chrom][start:end].seq)
		if 'N' in seq:
			continue
		for i in range(0, len(seq), 3):
			context = seq[i:(i+3)]
			if context in count_dict:
				count_dict[context] += 1
		valid_count += 1
	f.close()
	f = open(f"context_count.txt", "w")
	print("context", "count", sep = "\t", file = f)
	for key in count_dict.keys():
		print(key, count_dict[key], sep = '\t', file = f)
	f.close()

if __name__=="__main__":
	main()
