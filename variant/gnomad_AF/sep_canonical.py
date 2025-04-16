import re
import gzip
from os.path import exists

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

def to_num(x):
	if x == ".":
		return 0
	else:
		return(int(x))

def main():
	canonical_set = create_set("../prot/geneset_uniprot.txt")
	input_filename = "all_pos_alt_missense_sep_canonical_gnomad.txt"

	input_file = open(input_filename, "r")
	output_dir = "./mis_info_by_protein/"
	count = 0

	for line in input_file:
		if count % 1000000 == 0:
			print(str(count) + " variants processed.")
		count += 1
		line_split = line.strip().split("\t")
		chrom, pos, ref, alt, trans_id, aa_pos, aa, AN1, AC1, AN2, AC2, AN3, AC3, AN4, AC4 = (*line_split, )
		if trans_id in canonical_set:
			uniprot_id = canonical_set[trans_id]
		else:
			continue
		AN1 = to_num(AN1)
		AN2 = to_num(AN2)
		AN3 = to_num(AN3)
		AN4 = to_num(AN4)
		AC1 = to_num(AC1)
		AC2 = to_num(AC2)
		AC3 = to_num(AC3)
		AC4 = to_num(AC4)
		if (AN1 + AN2 + AN3 + AN4 == 0):
			continue
		aa_pos = aa_pos.split("/")[0]
		aa = aa.split("/")
		if len(aa) == 1:
			AA_ref = aa[0]
			AA_alt = aa[0]
		else:
			AA_ref = aa[0]
			AA_alt = aa[1]
		output_filename = output_dir + uniprot_id + "_info.txt"
		if exists(output_filename):
			output_file = open(output_filename, "a")
		else:
			output_file = open(output_filename, "w")
			print("Chrom\tPos\tRef\tAlt\tUniprotID\tTranscriptID\tAA_ref\tAA_alt\tTranscript_AA_pos\tgnomAD_NFE_AN\tgnomAD_NFE_AC\tgnomAD_AFR_AN\tgnomAD_AFR_AC", file = output_file)
		print(chrom, pos, ref, alt, uniprot_id, trans_id, AA_ref, AA_alt, aa_pos, AN1 + AN2, AC1 + AC2, AN3 + AN4, AC3 + AC4, sep = "\t", file = output_file)
		output_file.close()

	input_file.close()


if __name__ == "__main__":
	main()
