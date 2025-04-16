import re

def main():
	input_file = open("diff_pos_alt_lof.vcf", "r")

	lineperfile = 20000

	count = 0
	filecount = 0
	
	header = []
	
	for line in input_file:
		if re.search(r"^#", line):
			header.append(line)
			continue
		count += 1
		if count % lineperfile == 1:
			if count != 1:
				output_file.close()
			filecount += 1
			output_filename = "lof_partition/lof_" + str(filecount) + ".vcf"
			output_file = open(output_filename, "w")
			print(*header, sep = '', end = '', flush = True, file = output_file)
		print(line, file = output_file, end = '', flush = True)

	output_file.close()

	input_file.close()

if __name__ == '__main__':
	main()
