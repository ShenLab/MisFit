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

def to_int(x):
	if x == ".":
		return int(0)
	else:
		return int(x)

def to_float(x):
	if x == ".":
		return 0.0
	else:
		return float(x)

def main():
	basic_index = [i for i in range(7)]
	mu_index = [7, 8, 9]
	mu_name = ["gnomAD_mu_old", "gnomAD_mu", "roulette_mu"]
	mu_scaler = [1.787, 1.015e-7, 1.015e-7]
	pop_name = ["UKBB", "TOPMed", "gnomAD_NFE_genome", "gnomAD_NFE_exome", "gnomAD_AFR_genome", "gnomAD_AFR_exome", "AOU_EUR"]
	AN_index = [10, 12, 14, 16, 18, 20, 22]
	AC_index = [11, 13, 15, 17, 19, 21, 23]
	canonical_set = create_set("../prot/geneset_uniprot.txt")
	input_filename = "all_pos_alt_synonymous_sep_canonical_AF.txt.gz"

	input_file = gzip.open(input_filename, "rt")
	output_dir = "./syn_info_by_protein/"
	count = 0

	for line in input_file:
		if count % 1000000 == 0:
			print(str(count) + " variants processed.")
		count += 1
		line_split = line.strip().split("\t")

		chrom, pos, ref, alt, trans_id, aa_pos, aa = (*[line_split[i] for i in basic_index], )
		if trans_id in canonical_set:
			uniprot_id = canonical_set[trans_id]
		else:
			continue
		aa_pos = aa_pos.split("/")[0]
		aa = aa.split("/")
		if len(aa) == 1:
			AA_ref = aa[0]
			AA_alt = aa[0]
		else:
			AA_ref = aa[0]
			AA_alt = aa[1]

		mu = [to_float(line_split[i]) for i in mu_index]
		if not any(mu):
			continue
		for i in range(len(mu)):
			mu[i] = mu[i] * mu_scaler[i]
		
		AN = [to_int(line_split[i]) for i in AN_index]
		if not any(AN):
			continue
		AC = [to_int(line_split[i]) for i in AC_index]

		output_filename = output_dir + uniprot_id + "_info.txt"
		if exists(output_filename):
			output_file = open(output_filename, "a")
		else:
			output_file = open(output_filename, "w")
			print("Chrom\tPos\tRef\tAlt\tUniprotID\tTranscriptID\tAA_ref\tAA_alt\tTranscript_AA_pos\t", end = "", file = output_file)
			print(*mu_name, sep = "\t", end = "\t", file = output_file)
			print(*[i + "_AN" for i in pop_name], sep = "\t", end = "\t", file = output_file)
			print(*[i + "_AC" for i in pop_name], sep = "\t", end = "\n", file = output_file)
		print(chrom, pos, ref, alt, uniprot_id, trans_id, AA_ref, AA_alt, aa_pos, sep = "\t", end = "\t", file = output_file)
		print(*["{:.4e}".format(i) for i in mu], sep = "\t", end = "\t", file = output_file)
		print(*AN, sep = "\t", end = "\t", file = output_file)
		print(*AC, sep = "\t", end = "\n", file = output_file)
		output_file.close()

	input_file.close()


if __name__ == "__main__":
	main()
