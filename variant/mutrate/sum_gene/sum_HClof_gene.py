import re

def main():
	input_file = open("HClof_ukbbaf_gnomadmu.vcf", "r")

	summary = {}
	# count, an, ac, mu

	for line in input_file:
		if re.search(r"^#", line):
			continue
		info = line.strip().split("\t")[7]
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

	input_file.close()

	output_file = open("sum_lof_gene.txt", "w")
	for gene in summary:
		print(gene, *summary[gene], sep = "\t", file = output_file)
	output_file.close()

if __name__ == '__main__':
	main()
