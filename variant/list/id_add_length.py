import pandas as pd

def main():
	df0 = pd.read_table("geneset_uniprot.txt", sep = "\t")
	df1 = pd.read_table("uniprot_length.tsv.gz", sep = "\t")
	df1.columns = ['UniprotID', 'UniprotEntryName', 'Length']
	df = pd.merge(df0, df1, how = "left")
	df.to_csv("geneset_uniprot_len.txt", sep = "\t", index = False)

if __name__=="__main__":
	main()
