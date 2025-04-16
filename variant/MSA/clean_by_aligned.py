import pandas as pd
import re
import gzip
from Bio import pairwise2, SeqIO
from Bio.Align import substitution_matrices
from os.path import exists
import sys

def clean_seq(redundant_seq):
	clean_seq = ""
	positions = []
	for i in range(len(redundant_seq)):
		if redundant_seq[i] != '-':
			clean_seq += redundant_seq[i]
			positions.append(i)
	return positions, clean_seq

def extract_pos(seq0, redundant_seq1, seqA, seqB): # preserve all tokens in seq0, gives out list of corresponding position in seq1, None if mapped to gap
	redundant_seq1_pos, seq1 = clean_seq(redundant_seq1)
	if seq0 == seq1:
	    return redundant_seq1_pos
	posA, _ = clean_seq(seqA)
	posB, ori_seq1 = clean_seq(seqB)
	assert seq1 == ori_seq1
	extracted = [None] * len(posA)
	curr_index = 0
	for i, j in enumerate(posA):
		try:
			indexB = posB.index(j, curr_index)
			extracted[i] = redundant_seq1_pos[indexB]
			curr_index = indexB + 1
		except:
			pass
	return extracted

def extract_seq(seq, extracted_pos):
	new_seq = ['-'] * len(extracted_pos)
	for i, j in enumerate(extracted_pos):
		if j is not None:
			new_seq[i] = seq[j]
	return ''.join(new_seq)
	
def process_one_id(id0, seq_file, id1, MSA_file, out_file, iso_align_file): # by ensembl_protein_id
	records_list = []
	with gzip.open(seq_file, "rt") as f:
		record0 = SeqIO.read(f, "fasta")
	if exists(MSA_file):
		with gzip.open(MSA_file, "rt") as f:
			for record in SeqIO.parse(f, "fasta"):
				if record.id.split("|")[0] == id1:
					seq1  = record.seq
				else:
					records_list.append((record.id, record.seq))
		with open(iso_align_file, "r") as f:
			records = list(SeqIO.parse(f, "fasta"))
			seqA = records[0].seq
			seqB = records[1].seq
		extracted_pos = extract_pos(record0.seq, seq1, seqA, seqB)
	with gzip.open(out_file, "wt") as f:
		print(">" + id0, record0.seq, sep = "\n", file = f)
		for record in records_list:
			name, seq = record
			print(">" + name, extract_seq(seq, extracted_pos), sep = "\n", file = f)
	return len(records_list) + 1

def main(uniprot_id, gene_id):
	out_dir = "MSA_by_uniprot/"
	MSA_file = "Ensembl_genetree/" + gene_id + "_aligned.fasta.gz"
	seq_file = "uniprot_seq/" + uniprot_id + ".fasta.gz"
	out_file = out_dir + uniprot_id + "_aligned.fasta.gz"
	align_file = "iso_align/" + uniprot_id + "_pair.fasta"
	depth = process_one_id(uniprot_id, seq_file, gene_id, MSA_file, out_file, align_file)

if __name__=="__main__":
	uniprot_id = sys.argv[1]
	gene_id = sys.argv[2]
	main(uniprot_id, gene_id)



