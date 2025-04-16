import re
import gzip

def main():
	input_file = gzip.open("all_pos_alt_gnomad_AF_tag.vcf.gz", "rt")
	output_file = open("all_pos_alt_gnomad_AF.vcf", "w")
	for line in input_file:
		line = line.strip()
		if re.search("^#", line):
			print(line, file = output_file)
			continue
		line_split = line.split("\t")
		if line_split[6] == '.': # no flag
			print(line, file = output_file)
		elif line_split[7] == '.': # no AF
			print(line, file = output_file)
		else:
			all_pop = line_split[7].split(";")
			filter_pop = []
			rmgenome = re.search("NOGENOME", line_split[6])
			rmexome = re.search("NOEXOME", line_split[6])
			for pop in all_pop:
				pop_search_genome = re.search("genome", pop)
				pop_search_exome = re.search("exome", pop)
				if (pop_search_genome is not None) and (rmgenome is None):
					filter_pop.append(pop)
				elif (pop_search_exome is not None) and (rmexome is None):
					filter_pop.append(pop)
			if len(filter_pop) == 0:
				line_split[7] = '.'
			else:
				line_split[7] = ';'.join(filter_pop)
			print(*line_split, sep = "\t", file = output_file)
	input_file.close()
	output_file.close()

if __name__=="__main__":
	main()
