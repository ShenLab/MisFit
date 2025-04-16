from Bio import SeqIO
import pandas as pd
import gzip
import json
import numpy as np

with open("setting.json", "r") as f:
	setting = json.load(f)

def seq_to_int(seq):
	seq_int = [setting['AA_table'].get(aa, len(setting['AA_table'])) for aa in seq]
	return np.array(seq_int, dtype = np.int64)

def main():
	taxon_dict = {}
	with open("taxon_id_orthologue.txt", "r") as f:
		i = 1
		for line in f:
			taxon_dict[int(line.strip())] = i
			i += 1
	summary = open("orthologues_depth.txt", "w")
	print("UniprotID", "depth", sep = "\t", file = summary)

	geneset = pd.read_table("../list/geneset_uniprot.txt")
	for _, row in geneset.iterrows():
		GeneID = row['GeneID']
		UniprotID = row['UniprotID']
		genetree = {}
		with gzip.open("MSA_by_uniprot/" + UniprotID + "_aligned.fasta.gz", "rt") as f:
			for record in SeqIO.parse(f, "fasta"):
				genetree[record.id.split("|")[0]] = record.seq

		orig_seq = genetree[UniprotID]
		msa = np.full((len(orig_seq), len(taxon_dict) + 1), len(setting['AA_table']), dtype = np.int64)
		msa[:,0] = seq_to_int(orig_seq)
		cover = np.zeros((len(taxon_dict) + 1), dtype = np.float32)
		cover[0] = 1.
		ortho_table = pd.read_table("Ensembl_orthologues/" + GeneID + "_unorder.txt.gz")
		for _, ortho in ortho_table.iterrows():
			if ortho['gene_id'] in genetree:
				index = taxon_dict[ortho['taxon_id']]
				if cover[index] == 0:
					msa[:,index] = seq_to_int(genetree[ortho['gene_id']])
					cover[index] = 1

		np.save("MSA_orthologues/" + UniprotID + "_MSA.npy", msa)
		np.save("MSA_orthologues/" + UniprotID + "_cover.npy", cover)
		print(UniprotID, int(np.sum(cover)), sep = "\t", file = summary)
	summary.close()
if __name__=="__main__":
	main()
