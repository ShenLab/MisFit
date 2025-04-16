import re

def main():
	input_filename = "all_pos_alt_UKBB_AN.vcf"
	output_filename = "all_pos_alt_UKBB_AN.bed"
	input_file = open(input_filename, "r")
	output_file = open(output_filename, "w")
	max_inteval = 10
	prev_chrom = "0"
	prev_pos = 1
	start = 1
	mark_pos = start
	AN = 0
	
	for line in input_file:
		if re.search(r"^#", line):
			continue

		line_split = line.strip().split("\t")
		chrom = line_split[0]
		pos = int(line_split[1])

		# position not continuous: start another region record
		if (chrom != prev_chrom) or (pos - prev_pos > max_inteval):
			if (AN != 0) and (start < prev_pos):
				print(prev_chrom, start, prev_pos, AN, sep = "\t", file = output_file)
			start = pos
			mark_pos = start
			AN = 0

		prev_chrom = chrom
		prev_pos = pos

		current_search = re.search(r"UKBB_AN=(\d+)", line_split[7])
		if current_search:
			AN_new = current_search.group(1)
			if (AN != AN_new) and (AN != 0):
				print(chrom, start, mark_pos + (pos - mark_pos) // 2, AN, sep = "\t", file = output_file)
				start = mark_pos + (pos - mark_pos) // 2 + 1
			mark_pos = pos
			AN = AN_new

	if (AN != 0) and (start < prev_pos):
		print(chrom, start, pos, AN, sep = "\t", file = output_file)
				
	input_file.close()
	output_file.close()


if __name__ == '__main__':
	main()
