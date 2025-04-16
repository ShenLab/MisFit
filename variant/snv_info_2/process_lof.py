import pandas as pd
from os.path import exists

def main():
	munames = ["roulette_mu", "gnomAD_mu_old"]
	popnames = ["UKBB", "gnomAD_NFE_exome", "gnomAD_NFE_genome"]
	geneset = pd.read_table("../list/geneset_uniprot_len.txt")
	for _, row in geneset.iterrows():
		uniprot_id = row['UniprotID']
		filename = f"lof_info_by_protein/{uniprot_id}_info.txt.gz"
		if exists(filename):
			df = pd.read_table(filename)
			df['AN'] = 0
			df['AC'] = 0
			for popname in popnames:
				df['AN'] += df[popname + "_AN"]
				df['AC'] += df[popname + "_AC"]
			df['mu'] = 0.
			for muname in munames:
				df.loc[df['mu']==0, 'mu'] = df.loc[df['mu']==0, muname]
			df = df[(df['outlier']==False)&(df['mu']>0)&(df['AN']>0)][['AN', 'AC', 'mu']]
			if len(df) > 0:
				print(uniprot_id)
				df.to_csv(f"lof_ukbb_gnomad/{uniprot_id}_info.txt.gz", sep = "\t", index = False)

if __name__=="__main__":
	main()
