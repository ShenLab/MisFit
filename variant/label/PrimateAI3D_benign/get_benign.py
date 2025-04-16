import pandas as pd
from os.path import exists

def main():
	geneset = pd.read_table("geneset_uniprot_len.txt")
	for _, row in geneset.iterrows():
		uniprot_id = row['UniprotID']
		symbol = row['Symbol']
		if exists(f"PrimateAI3D/{symbol}.csv"):
			df = pd.read_csv(f"PrimateAI3D/{symbol}.csv")
		else:
			continue
		df = df[df['Primate Allele Count'] > 0]
		df = df[['Chromosome','Position','Reference','Alternate']]
		df.columns = ['Chrom', 'Pos', 'Ref', 'Alt']
		canon = pd.read_table(f'snv_only/{uniprot_id}.txt.gz')
		df = pd.merge(canon, df)
		if len(df) > 0:
			df.to_csv(f"mis_benign/{uniprot_id}.txt.gz", sep = "\t", index = False)

if __name__=="__main__":
	main()
