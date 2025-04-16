import re
import gzip

class genome_position:
	global all_chromosomes 
	all_chromosomes = [str(i) for i in range(1,23)] + ["MT", "X", "Y", ""]
	def __init__(self, chromosome, position):
		self.chromosome = str(chromosome)
		self.position = int(position)
	def __eq__(self, other):
		if (self.chromosome == other.chromosome) and (self.position == other.position):
			return True
		else:
			return False
	def __lt__(self, other):
		if (all_chromosomes.index(self.chromosome) < all_chromosomes.index(other.chromosome)):
			return True
		elif (all_chromosomes.index(self.chromosome) ==  all_chromosomes.index(other.chromosome)) and (self.position < other.position):
			return True
		else:
			return False
	def __gr__(self, other):
		if (all_chromosomes.index(self.chromosome) > all_chromosomes.index(other.chromosome)):
			return True
		elif (all_chromosomes.index(self.chromosome) ==  all_chromosomes.index(other.chromosome)) and (self.position > other.position):
			return True
		else:
			return False
	def __le__(self, other):
		return (self < other) or (self == other)
	def __ge__(self, other):
		return (self > other) or (self == other)

def process_region(region_file):
	region = region_file.readline()
	if region == "":
		return genome_position("", 0), genome_position("", 0)
	region_split = region.strip().split("\t")
	region_start = genome_position(region_split[0], region_split[1])
	region_end = genome_position(region_split[0], region_split[2])
	return region_start, region_end

def main():
	pos_file = gzip.open("/share/terra/Users/yz3419/annotation/database/UKBB/UKBB_hg38_SNV.recode.vcf.gz", "rt")
	region_file = open("exome_regions.txt", "r")
	output_file = open("exome_AN.bed", "w")
	
	region_start, region_end = process_region(region_file)
	curr_AN = None
	marker_pos = None

	for line in pos_file:
		if re.search(r"^#", line):
			continue
		line_split = line.strip().split("\t")
		AN_pos = genome_position(line_split[0], line_split[1])
		while region_end <= AN_pos:
			if curr_AN is not None:
				print(region_start.chromosome, str(region_start.position), str(region_end.position), curr_AN, sep = "\t", file = output_file, flush = True)
			region_start, region_end = process_region(region_file)
			curr_AN = None
			marker_pos = None
		if region_start > AN_pos:
			continue
		AN = re.search(r"UKBB_AN=(\d+)", line_split[7]).group(1)
		if (curr_AN is not None) and (curr_AN != AN):
			print(region_start.chromosome, str(region_start.position), str((marker_pos + AN_pos.position + 1) // 2), curr_AN, sep = "\t", file = output_file, flush = True)
			region_start.position = (marker_pos + AN_pos.position + 1) // 2
		marker_pos = AN_pos.position
		curr_AN = AN
	if curr_AN is not None:
		print(region_start.chromosome, str(region_start.position), str(region_end.position), curr_AN, sep = "\t", file = output_file, flush = True)

	pos_file.close()
	region_file.close()
	output_file.close()


if __name__=="__main__":
	main()

