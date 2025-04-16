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
	taxon_df = pd.read_table("overview.table.tsv")
	all_taxon_id = taxon_df['Species Taxonomy ID'].unique()
	all_taxon_id.sort()
	taxon_dict = {}
	with open("zoonomia_taxon_id.txt", "w") as f:
		for i in range(len(all_taxon_id)):
			taxon_id = all_taxon_id[i]
			print(taxon_id, file = f)
			taxon_dict[taxon_id] = i + 1
	assembly_dict = {}
	for _, row in taxon_df.iterrows():
		name = row['Assembly name']
		taxon_id = row['Species Taxonomy ID']
		assembly_dict[f"vs_{name}"] = taxon_dict[taxon_id]

	summary = open("orthologues_depth.txt", "w")
	print("UniprotID", "depth", sep = "\t", file = summary)

	geneset = pd.read_table("../list/geneset_uniprot.txt")
	for _, row in geneset.iterrows():
		UniprotID = row['UniprotID']
		msa = None
		with gzip.open("MSA_by_uniprot/" + UniprotID + "_aligned.fasta.gz", "rt") as f:
			for record in SeqIO.parse(f, "fasta"):
				if record.id == UniprotID:
					msa = np.full((len(record.seq), len(taxon_dict) + 1), len(setting['AA_table']), dtype = np.int64)
					msa[:,0] = seq_to_int(record.seq)
					cover = np.zeros((len(taxon_dict) + 1), dtype = np.float32)
					cover[0] = 1.
					continue
				index = assembly_dict[record.id]
				if cover[index] == 0:
					msa[:,index] = seq_to_int(record.seq)
					cover[index] = 1
		np.save("MSA_orthologues/" + UniprotID + "_MSA.npy", msa)
		np.save("MSA_orthologues/" + UniprotID + "_cover.npy", cover)
		print(UniprotID, int(np.sum(cover)), sep = "\t", file = summary)
	summary.close()

if __name__=="__main__":
	main()
