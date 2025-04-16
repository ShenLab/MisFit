import re

def main():
	output_file = open("diff_pos_alt_loftee.vcf", "w")

	for i in range(1, 33):
		input_filename = "lof_partition/loftee_" + str(i) + ".vcf"
		input_file = open(input_filename, "r")
		for line in input_file:
			if re.search(r"^#", line):
				if i == 1:
					print(line, end = '', file = output_file)
				else:
					continue
			else:
				print(line, end = '', file = output_file)
		input_file.close()

	output_file.close()

if __name__ == '__main__':
	main()
