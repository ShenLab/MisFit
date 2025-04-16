import re
import gzip

def main():
	uniprot_file = open("alphafold2_uniprotid.txt", "r")
	uniprot_ids = set()
	for line in uniprot_file:
		uniprot_ids.add(line.strip())
	uniprot_file.close()
	
	mapping_file = gzip.open("HUMAN_9606_idmapping.dat.gz", "rt")
	output_file = open("alphafold2_ensembltrans_all.txt", "w")
	for line in mapping_file:
		line_split = line.strip().split("\t")
		if (line_split[0] in uniprot_ids) and (line_split[1] == "Ensembl_TRS"):
			print(line_split[0], line_split[2].split(".")[0], sep = "\t", file = output_file)
	mapping_file.close()
	output_file.close()

if __name__=="__main__":
	main()
