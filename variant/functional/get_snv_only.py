import pandas as pd

def main():
	geneset = pd.read_table("../list/geneset_uniprot.txt")
	for _, row in geneset.iterrows():
		uniprot_id = row['UniprotID']
		df = pd.read_table(f"scores/{uniprot_id}_scores.txt.gz")
		df = df.rename(columns = {'Protein_position': 'Ensembl_protein_position'})
		df[['Chrom', 'Pos', 'Ref', 'Alt', 'Ensembl_protein_position', 'Uniprot_position', 'AA_ref', 'AA_alt']].to_csv(f"snv_only/{uniprot_id}.txt.gz", sep = "\t", index = False)

if __name__=="__main__":
	main()
