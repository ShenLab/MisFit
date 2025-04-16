import requests
import gzip
import sys
import pandas as pd


def parse_one(GeneID):
	url = "https://may2024.rest.ensembl.org/homology/id/homo_sapiens/" + GeneID + "?content-type=application/json;compara=vertebrates;sequence=none;type=orthologues"
	r = requests.get(url)
	
	if not r.ok:
		return None
	try:
		data = r.json()['data'][0]['homologies']
	except:
		return None
	records = {'gene_id': [], 'protein_id': [], 'species':[], 'taxon_id': []}
	for homology in data:
		target = homology['target']
		records['taxon_id'].append(target['taxon_id'])
		records['protein_id'].append(target['protein_id'])
		records['gene_id'].append(target['id'])
		records['species'].append(target['species'])
	return records
	
def main(start, end):
	all_gene_id = []
	with open("../list2/geneset.txt", "r") as f:
#	with open("fail.txt", "r") as f:
		line = f.readline()
		for line in f:
			all_gene_id.append(line.strip().split("\t")[0])
	for GeneID in all_gene_id[start:end]:
		records = parse_one(GeneID)
		if records is None:
			with open("failo.txt", "a") as fail_log:
				print(GeneID, file = fail_log)
			continue
		pd.DataFrame(records).to_csv("Ensembl_orthologues/" + GeneID + "_unorder.txt.gz", sep = "\t", index = False)
	print(f"{start} to {end} processed.")

if __name__=="__main__":
	start = int(sys.argv[1])
	end = int(sys.argv[2])
	main(start, end)

