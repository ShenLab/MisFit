import pandas as pd
import re
import gzip
from Bio import pairwise2, SeqIO
from Bio.Align import substitution_matrices
from os.path import exists
import sys

#matrix = substitution_matrices.load('BLOSUM62')

def clean_seq(redundant_seq):
	clean_seq = ""
	positions = []
	for i in range(len(redundant_seq)):
		if redundant_seq[i] != '-':
			clean_seq += redundant_seq[i]
			positions.append(i)
	return positions, clean_seq

def extract_pos(seq0, redundant_seq1): # preserve all tokens in seq0, gives out list of corresponding position in seq1, None if mapped to gap
	redundant_seq1_pos, seq1 = clean_seq(redundant_seq1)
	if seq0 == seq1:
	    return redundant_seq1_pos
#	align = pairwise2.align.globalds(seq0, seq1, matrix, -10., -0.5, one_alignment_only = True)[0]
	align = pairwise2.align.globalms(seq0, seq1, 1, -1, -3.5, 0., one_alignment_only = True)[0]
	seqA = align.seqA
	seqB = align.seqB
	posA, _ = clean_seq(seqA)
	posB, _ = clean_seq(seqB)
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
	
def process_one_id(id0, seq_file, id1, MSA_file, out_file): # by ensembl_protein_id
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
		extracted_pos = extract_pos(record0.seq, seq1)
	with gzip.open(out_file, "wt") as f:
		print(">" + id0, record0.seq, sep = "\n", file = f)
		for record in records_list:
			name, seq = record
			print(">" + name, extract_seq(seq, extracted_pos), sep = "\n", file = f)
	return len(records_list) + 1

def main(filename, start, end):
	geneset_df = pd.read_table(filename)
	out_dir = "MSA_by_uniprot/"
	for _, row in geneset_df.iloc[start:end].iterrows():
		uniprot_id = row['UniprotID']
		gene_id = row['GeneID']
		MSA_file = "Ensembl_genetree/" + gene_id + "_aligned.fasta.gz"
		seq_file = "uniprot_seq/" + uniprot_id + ".fasta.gz"
		out_file = out_dir + uniprot_id + "_aligned.fasta.gz"
		try:
			depth = process_one_id(uniprot_id, seq_file, gene_id, MSA_file, out_file)
		except:
			with open("log.txt", "a") as f:
				print(gene_id, file = f)

if __name__=="__main__":
	filename = sys.argv[1]
	start = int(sys.argv[2])
	end = int(sys.argv[3])
	main(filename, start, end)



