import gzip
import re

def main():
	padding = 6
	all_chroms = [str(x) for x in range(1, 23)] + ["X", "Y"]
	f = gzip.open("exome_lcr.vcf.gz", "rt")
	o = gzip.open("exome_lcr_region.txt.gz", "wt")
	print("Chrom", "Start", "End", "GeneID", sep = "\t", file = o)
	chrom, start, end, gene = "0", 0, 0, set()
	for line in f:
		if re.search(r"^#", line):
			continue
		line_split = line.strip().split()
		curr_chrom = line_split[0][3:]
		curr_pos = int(line_split[1])
		if (curr_chrom ==  chrom) and (curr_pos <= end):
			continue
		if (curr_chrom != chrom) or (curr_pos > end + padding * 2):
			if chrom in all_chroms:
				for each_gene in gene:
					print(chrom, start - padding, end + padding, each_gene, sep = "\t", file = o)
			chrom = curr_chrom
			start = curr_pos
			end = curr_pos
			gene = set()
		else:
			end = curr_pos
		csq_search = re.search(r"vep=([^;]+)", line_split[7])
		if csq_search:
			all_csq = csq_search.group(1).split(",")
			for csq in all_csq:
				curr_gene = csq.split("|")[4]
				if curr_gene != "":
					gene.add(curr_gene)
		
	f.close()
	o.close()

if __name__=="__main__":
	main()

