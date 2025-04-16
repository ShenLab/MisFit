import gzip
import re

def main():
	input_file = gzip.open("all_pos_alt_genome_exclude.vcf.gz", "rt")
	output_file = open("all_pos_alt_genome_exclude.vcf", "w")
	curr_pos = ""
	for line in input_file:
		line = line.strip()
		if re.search("^#", line):
			print(line, file = output_file)
			continue
		line_split = line.split("\t")
		chrom = line_split[0]
		pos = line_split[1]
		if pos == curr_pos:
			continue
		ref = line_split[3]
		for alt in ["A", "C", "G", "T"]:
			if ref == alt:
				continue
			print(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tNOGENOME\t.", file = output_file)
	input_file.close()
	output_file.close()

if __name__=="__main__":
	main()
