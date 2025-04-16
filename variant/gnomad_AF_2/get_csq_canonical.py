import re
import gzip

def create_set(input_filename, header = True, key_column = 2, map_column = 0):
    input_file = open(input_filename, "r")
    if header:
        line = input_file.readline()
    canonical_set = {} 
    for line in input_file:
        line_split = line.strip().split("\t")
        canonical_set[line_split[key_column]] = line_split[map_column]
    input_file.close()
    return canonical_set

def process_csq(csq, canonical_set):
	fields = csq.split("|")
	if fields[6] in canonical_set:
		# Feature field corresponding to transcript_id --> mao to uniprot
		return canonical_set[fields[6]]
	else:
		return None

def main():
	canonical_set = create_set("../list/geneset_uniprot.txt")
	input_filename = "all_pos_alt_exome_exclude.vcf.gz"
	output_dir = "exome_exclude/"

	input_file = gzip.open(input_filename, "rt")

	count = 0

	for line in input_file:
		if re.search(r"^#", line):
			continue
		if count % 100000 == 0:
			print(str(count) + " variants processed.")
		count += 1
		line_split = line.strip().split("\t")
		csq_all = re.search(r"CSQ=(.*)$", line_split[7]).group(1).split(",")
		for csq in csq_all:
			uniprot_id = process_csq(csq, canonical_set)
			if uniprot_id is not None:
				output_file = open(f"{output_dir}/{uniprot_id}.txt", "a")
				print(line_split[0], line_split[1], line_split[3], line_split[4], sep = "\t", file = output_file)
				output_file.close()

	input_file.close()


if __name__ == "__main__":
	main()
