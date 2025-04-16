import gzip
import re
import pandas as pd

def main():
	geneset = pd.read_table("geneset_uniprot_len.txt")
	trans_set = geneset['TranscriptID'].unique()
	f = gzip.open("gnomad_qc_csq.vcf.gz", "rt")
	for line in f:
		if re.search(r"^#", line):
			continue
		line_split = line.strip().split()
		csq_search = re.search(r"CSQ=([^;]+)", line_split[7])
		if csq_search:
			all_csq = csq_search.group(1).split(",")
			for csq in all_csq:
				gene = csq.split("|")[6]
				if gene in trans_set:
					filename = f"snv_qc/{gene}.txt"
					with open(filename, "a") as out:
						print(line_split[0], line_split[1], line_split[3], line_split[4], line_split[6], sep = "\t", file = out)
		
	f.close()

if __name__=="__main__":
	main()

