import pandas as pd
from os.path import exists

def main():
	ratio = 0.1
	sample_df = pd.DataFrame()
	geneset = pd.read_table("../variant/list/geneset_uniprot.txt")
	for i, row in geneset.iterrows():
		uniprot_id = row['UniprotID']
		filename = f"../variant/snv_info_2/syn_info_by_protein_mapping/{uniprot_id}_info.txt.gz"
		if exists(filename):
			df = pd.read_table(filename)
			df['UniprotID'] = uniprot_id
			sample = df.sample(frac = ratio, random_state = i)
			sample = sample[sample['roulette_mu'] > 1e-7]
			sample_df = pd.concat([sample_df, sample])
	sample_df.to_csv("syn_sample_highmu.txt.gz", compression = "gzip", index = False, sep = "\t")

if __name__=="__main__":
	main()