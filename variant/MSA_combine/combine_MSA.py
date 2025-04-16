import pandas as pd
import numpy as np

def main():
	taxon_1 = pd.read_table("taxon_1.txt", names = ['taxon'])
	taxon_1['source'] = "ensembl"
	species_cover_1 = pd.read_table("species_cover_1.txt", names = ['cover'])
	taxon_1['cover'] = species_cover_1['cover']
	taxon_2 = pd.read_table("taxon_2.txt", names = ['taxon'])
	taxon_2['source'] = "zoonomia"
	species_cover_2 = pd.read_table("species_cover_2.txt", names = ['cover'])
	taxon_2['cover'] = species_cover_2['cover']
	taxon = pd.concat([taxon_1, taxon_2], ignore_index = True)
	taxon['index'] = [i for i in range(1, len(taxon) + 1)]
	taxon = taxon[taxon['cover'] > 10000]
	taxon = taxon.sort_values(by = ['cover'], ascending = False)
	taxon = taxon.drop_duplicates(subset = ['taxon'])
	taxon = taxon.sort_values(by = ['taxon'])
	taxon_index = taxon['index'].to_numpy()
	taxon_index = np.insert(taxon_index, 0, 0)
	taxon[['taxon', 'source', 'cover']].to_csv("taxon_id_combined.txt", sep = "\t", index = False)
	geneset = pd.read_table("../list/geneset_uniprot.txt")
	for _, row in geneset.iterrows():
		uniprot_id = row['UniprotID']
		MSA_1 = np.load(f"MSA_1/{uniprot_id}_MSA.npy")
		cover_1 = np.load(f"MSA_1/{uniprot_id}_cover.npy")
		MSA_2 = np.load(f"MSA_2/{uniprot_id}_MSA.npy")
		cover_2 = np.load(f"MSA_2/{uniprot_id}_cover.npy")
		MSA_combine = np.concatenate((MSA_1, MSA_2[:,1:]), axis = 1)
		cover_combine = np.concatenate((cover_1, cover_2[1:]))
		np.save(f"MSA_orthologues/{uniprot_id}_MSA.npy", MSA_combine[:, taxon_index])
		np.save(f"MSA_orthologues/{uniprot_id}_cover.npy", cover_combine[taxon_index])

if __name__=="__main__":
	main()
