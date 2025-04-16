import pandas as pd

def main():
	geneset = pd.read_table("summary.txt")
	all_df = []
	for uniprot_id in geneset['UniprotID'].tolist():
		df = pd.read_table(f"var_by_uniprot_function/{uniprot_id}.txt.gz")
		all_df.append(df)
	combined = pd.concat(all_df, axis = 0)
	combined.to_csv("clinvar_missense.txt.gz", sep = "\t", index = False)

if __name__=="__main__":
	main()
