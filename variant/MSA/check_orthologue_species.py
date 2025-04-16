import pandas as pd

def main():
	geneset = pd.read_table("../list/geneset_uniprot.txt")
	species_set = set()
	for _, row in geneset.iterrows():
		GeneID = row['GeneID']
		ortho = pd.read_table("Ensembl_orthologues/" + GeneID + "_unorder.txt.gz")
		for _, taxon_id in ortho['taxon_id'].items():
			species_set.add(taxon_id)
	species_list = list(species_set)
	species_list.sort()
	with open("taxon_id_orthologue.txt", "w") as f:
		print(*species_list, sep = "\n", file = f)

if __name__=="__main__":
	main()
