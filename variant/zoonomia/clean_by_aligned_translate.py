import pandas as pd
import re
import gzip
from Bio import pairwise2, SeqIO
import os
import sys

def translate(dna_seq, codon_map):
	aa_seq = ["X"] * (len(dna_seq) // 3)
	for i in range(len(aa_seq)):
		codon = dna_seq[3 * i:(3 * i + 3)]
		if codon in codon_map:
			aa_seq[i] = codon_map[codon]
	return "".join(aa_seq)

def clean_seq(redundant_seq):
	positions = []
	for i in range(len(redundant_seq)):
		if (redundant_seq[i] != '-') and (redundant_seq[i] != 'X'):
			positions.append(i)
	return positions

def extract_seq(seq, extracted_pos):
	new_seq = ['-'] * len(extracted_pos)
	for i, j in enumerate(extracted_pos):
		if j is not None:
			new_seq[i] = seq[j]
	return "".join(new_seq)

def process_one_id(codon_file, aa_file, codon_map): # by transcript id and gene name
	records_list = {}
	with gzip.open(codon_file, "rt") as f:
		for record in SeqIO.parse(f, "fasta"):
			if record.id in records_list:
				continue
			record_aa_seq = translate(record.seq, codon_map)
			records_list[record.id] = record_aa_seq
		extracted_pos = clean_seq(records_list["REFERENCE"])
	with gzip.open(aa_file, "wt") as f:
		seq = records_list.pop("REFERENCE")
		print(">REFERENCE", extract_seq(seq, extracted_pos), sep = "\n", file = f)
		for name in records_list.keys():
			seq = records_list[name]
			print(">" + name, extract_seq(seq, extracted_pos), sep = "\n", file = f)
	return len(records_list) + 1

def process_codon(codon_filename, codon_column, aa_column):
	codon_map = {}
	with open(codon_filename) as f:
		f.readline()
		for line in f:
			line_split = line.strip().split("\t")
			codon_map[line_split[codon_column]] = line_split[aa_column]
	codon_map['---'] = '-'
	return codon_map

def main():
	codon_map = process_codon("codon_table.txt", 0, 2)
	codon_dir = "MultiCodon/"
	aa_dir = "MSA_by_transcript/"
	for filename in os.listdir(codon_dir):
		print(filename.split(".")[0])
		process_one_id(f"{codon_dir}/{filename}", f"{aa_dir}/{filename}", codon_map)


if __name__ == "__main__":
	main()


