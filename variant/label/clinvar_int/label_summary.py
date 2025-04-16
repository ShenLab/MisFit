import pandas as pd
from os.path import exists

def main():
	geneset = pd.read_table("../../list/geneset_uniprot.txt")
	summary = open("summary_pop.txt", "w")
	print("UniprotID\tpositive\tnegative", file = summary)
	for _, row in geneset.iterrows():
		uniprot_id = row['UniprotID']
		filename = f"var_by_uniprot_pop/{uniprot_id}.txt.gz"
		if exists(filename):
			df = pd.read_table(filename)
			positive = len(df[df['Label'] == 1])
			negative = len(df[df['Label'] == 0])
			print(f"{uniprot_id}\t{positive}\t{negative}", file = summary)
if __name__=="__main__":
	main()
