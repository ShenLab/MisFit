import json
from Bio import SeqIO
import gzip
import pandas as pd
import numpy as np
from os.path import exists

with open("setting.json", "r") as f:
	settings = json.load(f)
A = len(settings['AA_table']) + 1 # Amino Acid dictionary size + gap

def seq_to_list(seq):
	return [settings['AA_table'].get(AA, A-1) for AA in seq]

def aligned_to_tensor(fasta_filename):
	seq_list = []
	with gzip.open(fasta_filename, "rt") as f:
		for record in SeqIO.parse(f, "fasta"):
			seq_list.append(seq_to_list(record.seq))
	return  np.transpose(np.array(seq_list, dtype = np.int32))

def main():
	MSA_dir = "MSA_by_uniprot/"
	output_dir = "MSA_genetree/"
	df = pd.read_table("../list/geneset_uniprot_len.txt")
	for index, row in df.iterrows():
		uniprot_id = row['UniprotID']
		length = row['Length']
		fasta_filename = f'{MSA_dir}/{uniprot_id}_aligned.fasta.gz'
		tensor = aligned_to_tensor(fasta_filename)
		if tensor.shape[0] != length:
			print(f"{uniprot_id} length not match: MSA {tensor.shape[0]}, Uniprot {length}")
			continue

		filename = f'{output_dir}/{uniprot_id}_MSA.npy'
		np.save(filename, tensor)

		if index % 1000 == 0:
			print(f"{index} genes processed.")

if __name__=="__main__":
	main()

