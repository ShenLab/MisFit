import pandas as pd
import os
import re
import gzip
from Bio import pairwise2, SeqIO
import json

def positional_mapping(uniprot_id, transcript_id):
	fasta_filename = "../pep/unmapped/" + uniprot_id + ".fasta.gz"
	if not os.path.exists(fasta_filename):
		return None, None
	with gzip.open(fasta_filename, "rt") as f:
		records = list(SeqIO.parse(f, "fasta"))
	align = pairwise2.align.localms(records[0].seq, records[1].seq, 1, -1, -3.5, 0., one_alignment_only = True)[0]
	uniprot_seq = align.seqA
	ensembl_seq = align.seqB
	uniprot_pos = 0
	ensembl_pos = 0
	forward_mapping = {}
	reverse_mapping = {}
	for i in range(len(uniprot_seq)):
		if uniprot_seq[i]!='-':
			uniprot_pos += 1
		if ensembl_seq[i]!='-':
			ensembl_pos += 1
			if (align.start <= i) and (align.end > i) and (uniprot_seq[i] == ensembl_seq[i]):
				forward_mapping[ensembl_pos] = uniprot_pos
				reverse_mapping[uniprot_pos] = ensembl_pos
	return {"ID": uniprot_id, "mapping": forward_mapping}, {"ID": transcript_id, "mapping": reverse_mapping}

def main():
	geneset = pd.read_table("../list/geneset_uniprot.txt")
	
	for _, row in geneset.iterrows():
		transcript_id = row['TranscriptID']
		uniprot_id = row['UniprotID']
		forward_mapping, reverse_mapping = positional_mapping(uniprot_id, transcript_id)
		if forward_mapping is not None:
			with open("ensembl_to_uniprot_pos/" + transcript_id + ".json", "w") as f:
				json.dump(forward_mapping, f, indent = 4)
			with open("uniprot_to_ensembl_pos/" + uniprot_id + ".json", "w") as f:
				json.dump(reverse_mapping, f, indent = 4)

if __name__=="__main__":
	main()
