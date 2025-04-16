import re
import gzip

def create_set(input_filename, header = True, column = 1):
    input_file = open(input_filename, "r")
    if header:
        line = input_file.readline()
    canonical_set = set() 
    for line in input_file:
        line_split = line.strip().split("\t")
        canonical_set.add(line_split[column])
    input_file.close()
    return canonical_set

def process_csq(csq, canonical_set):
    fields = csq.split("|")
    if fields[6] in canonical_set:
        # Feature field corresponding to transcript_id 
        return fields
    else:
        return None

def get_content(search, default = ""):
	if search:
		return search.group(1)
	else:
		return default


def main():
	canonical_set = create_set("../prot/geneset_uniprot.txt", column = 2)
	input_filename = "../UKBB_AF/all_coding_alt_rawcsq_AF.vcf.gz"

	input_file = gzip.open(input_filename, "rt")
	output_dir = "./lgd_info_by_protein/"
	count = 0

	for line in input_file:
		if re.search(r"^#", line):
			continue
		if count % 1000000 == 0:
			print(str(count) + " variants processed.")
		count += 1
		line_split = line.strip().split("\t")
		csq_all = re.search(r"CSQ=([^;]*)", line_split[7]).group(1).split(",")
		mu_search = re.search(r"gnomAD_mu=([^;]*)", line_split[7])
		AN_search = re.search(r"UKBB_AN=([^;]*)", line_split[7])
		AC_search = re.search(r"UKBB_AC=([^;]*)", line_split[7])
		for csq in csq_all:
			all_fields = process_csq(csq, canonical_set)
			if all_fields is not None:
				trans_id = all_fields[6]
				if re.search("stop_gained|stop_lost|start_lost|frameshift|splice_acceptor|splice_donor", all_fields[1]): # LGD
					output_file = open(output_dir + trans_id + "_info.txt", "a")
					print(line_split[0], line_split[1], line_split[3], line_split[4], get_content(mu_search), get_content(AN_search), get_content(AC_search, "0"), sep = "\t", file = output_file)
					output_file.close()

	input_file.close()


if __name__ == "__main__":
	main()
