import re
import gzip
import pandas as pd

def process_gff(line):
	line_split = line.strip().split("\t")
	chrom = line_split[0]
	feature = line_split[2]
	start = line_split[3]
	end = line_split[4]

	if (feature == "CDS"):
		return chrom, start, end
	else:
		return None, None, None

def main():
	# inputs
	annot_filename = "gencode.v38.basic.annotation.gff3.gz"
	annot_file = gzip.open(annot_filename, "rt")

	# outputs
	pos_filename = "all_cds_regions.txt"
	pos_file = open(pos_filename, "w")

	# split line
	count = 0
	for line in annot_file:
		if re.search(r"^#", line):
			continue
		chrom, start, end = process_gff(line)
		if chrom is None:
			continue
		# to allow capturing of splice_donor and splice_acceptor
		start = int(start) - 2
		end = int(end) + 2
		print(f"{chrom}\t{start}\t{end}", file = pos_file)

	annot_file.close()
	pos_file.close()

if __name__ == "__main__":
	main()
