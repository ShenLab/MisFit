import requests, sys
#import pandas as pd
import gzip
import sys

def retrieve_tree(tree, records):
	if 'children' in tree:
		for child in tree['children']:
			retrieve_tree(child, records)
	else:
		seq = tree['sequence']['mol_seq']['seq']
		tax = tree['taxonomy']['scientific_name']
		seqid = tree['id']['accession']
		records.append((seqid, tax, seq))

def parse_one(GeneID):
	url = "https://may2021.rest.ensembl.org/genetree/member/id/" + GeneID + "?content-type=application/json;aligned=1"
	r = requests.get(url)
	
	if not r.ok:
		return None
	tree = r.json()['tree']
	records = []
	retrieve_tree(tree, records)
	return records
	
def main(start, end):
	all_gene_id = []
	fail_log = open("fail2.txt", "a")
	with open("/home/nas-0-1/nova.home/yz3419/data/variant/list/geneset_uniprot.txt", "r") as f:
#	with open("/home/nas-0-1/nova.home/yz3419/data/variant/MSA/fail.txt", "r") as f:
		for line in f:
			all_gene_id.append(line.strip().split("\t")[1])
#			all_gene_id.append(line.strip().split("\t")[0])
	for GeneID in all_gene_id[start:end]:
		records = parse_one(GeneID)
		if records is None:
			print(GeneID, file = fail_log)
			continue
		with gzip.open("/home/nas-0-1/nova.home/yz3419/data/variant/MSA/Ensembl_genetree/" + GeneID + "_aligned.fasta.gz", "wt") as f:
			for record in records:
				seqid, tax, seq = record
				print(">" + seqid + "|" + tax, file = f)
				print(seq, file = f)
	print(f"{start} to {end} processed.")
	fail_log.close()

if __name__=="__main__":
	start = int(sys.argv[1])
	end = int(sys.argv[2])
	main(start, end)

