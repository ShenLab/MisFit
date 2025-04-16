import gzip

def main():
	input_file = gzip.open("AOU_af_cds.txt.gz", "rt")
	output_file = open("AOU_af_cds_eur.txt", "w")
	chrom_dict = {}
	for chrom in [str(i) for i in range(1, 23)] + ["X", "Y"]:
		chrom_dict["chr" + chrom] = chrom
	print("Chrom\tPos\tRef\tAlt\tAN_AOU_eur\tAC_AOU_eur", file = output_file)
	input_file.readline()
	for line in input_file:
		line_strip = line.strip().split("\t")
		if line_strip[0] in chrom_dict:
			line_strip[0] = chrom_dict[line_strip[0]]
			print(*line_strip[0:6], sep = "\t", file = output_file)
		else:
			continue
	input_file.close()
	output_file.close()

if __name__=="__main__":
	main()
