from os.path import exists
import pandas as pd

def main():
	geneset = pd.read_table("../list/geneset_uniprot.txt")
	uniprot_set = []
	for _, row in geneset.iterrows():
		uniprot_id = row['UniprotID']
		if exists("mis_info_uniprot_ukbb/" + uniprot_id + "_uniaa.txt.gz"):
			uniprot_set.append(uniprot_id)
	with open("ukbb_covered_list.txt", "w") as f:
		print(*uniprot_set, sep = "\n", file = f)

if __name__=="__main__":
	main()
