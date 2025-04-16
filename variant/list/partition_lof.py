
def main():
	input_file = open("all_pos_alt_lof.vcf", "r")
	header = input_file.readline()
	input_file.readline()
	input_file.readline()
	header = header + input_file.readline()

	lineperfile = 20000

	count = 0
	filecount = 0
	
	for line in input_file:
		count += 1
		if count % lineperfile == 1:
			if count != 1:
				output_file.close()
			filecount += 1
			output_filename = "lof_partition/lof_" + str(filecount) + ".vcf"
			output_file = open(output_filename, "w")
			print(header, sep = '', end = '', flush = True, file = output_file)
		print(line, file = output_file, end = '', flush = True)

	output_file.close()

	input_file.close()

if __name__ == '__main__':
	main()