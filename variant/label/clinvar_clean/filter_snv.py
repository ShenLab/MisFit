
def main():
	input_file = open("clinvar_compare.diff.sites_in_files", "r")
	output_file = open("clinvar_compare_snv.vcf", "a")
	input_file.readline()
	NUC = ["C", "T", "A", "G"]
	for line in input_file:
		line_split = line.strip().split()
		if line_split[3] == 'B': # in file
			recent = 0
		elif line_split[3] == '1':
			recent = 1
		else:
			continue
		if not ((line_split[4] in NUC) and (line_split[6] in NUC)):
			continue
		print(line_split[0], line_split[1], '.', line_split[4], line_split[6], '.', '.', f"RECENT={recent}", sep = "\t", file = output_file)
			
	input_file.close()
	output_file.close()

if __name__=="__main__":
	main()
