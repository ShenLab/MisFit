import re

def main():
	input_file = open("all_pos_alt_synonymous_sep_canonical_ukbbaf_gnomadmu.vcf", "r")

	summary = {}
	# count, an, ac, mu
	all_count = 0
	all_an = 0
	all_ac = 0
	all_mu = 0.
	all_autochrom = [str(i) for i in range(1, 23)]

	prev_chrom = "0"
	prev_pos = "0"
	prev_alt = ""

	for line in input_file:
		if re.search(r"^#", line):
			continue
		line_split = line.strip().split("\t")
		chrom = line_split[0]
		pos = line_split[1]
		alt = line_split[4]
		info = line_split[7]
		mu_search = re.search(r"gnomAD_mu=([0-9\-e\.]+)", info)
		if mu_search:
			an_search = re.search(r"UKBB_AN=(\d+)", info)
			if an_search:
				an = int(an_search.group(1))
				mu = float(mu_search.group(1))
				ac_search = re.search(r"UKBB_AC=(\d+)", info)
				if ac_search:
					ac = int(ac_search.group(1))
				else:
					ac = 0
				csq = re.search(r"CSQ=([^;]+)", info).group(1)
				gene = csq.split("|")[4]
				if not (gene in summary):
					summary[gene] = [0, 0, 0, 0]
				summary[gene][0] += 1
				summary[gene][1] += an
				summary[gene][2] += ac
				summary[gene][3] += mu
				
				if ((chrom != prev_chrom) or (pos != prev_pos) or (alt != prev_alt)) and (chrom in all_autochrom):
					all_count += 1
					all_an += an
					all_ac += ac
					all_mu += mu
					if all_count % 500000 == 0:
						print(f"count: {all_count}; an: {all_an}; ac: {all_ac}; mu: {all_mu}", flush = True)

		prev_chrom = chrom
		prev_pos = pos
		prev_alt = alt

	input_file.close()
	print(f"count: {all_count}; an: {all_an}; ac: {all_ac}; mu: {all_mu}", flush = True)

	output_file = open("sum_syn_gene.txt", "w")
	for gene in summary:
		print(gene, *summary[gene], sep = "\t", file = output_file)
	output_file.close()

if __name__ == '__main__':
	main()
