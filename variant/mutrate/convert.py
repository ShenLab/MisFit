from pyliftover import LiftOver
import gzip

def add_chr(chrom):
	if chrom=="MT":
		return "chrM"
	else:
		return "chr" + str(chrom)

def rem_chr(chrom):
	chrom = chrom[3:]
	if chrom=="M":
		return "MT"
	else:
		return chrom

def main():
	lo = LiftOver('hg19', 'hg38')
	f = gzip.open("methylation.txt.gz", "rt")
	line = f.readline()

	output = open("methylation_hg38.txt", "w", 1)
	print("#CHROM\tPOS\tMETH", file = output)

	for line in f:
		line_split = line.strip().split("\t")
		locus = line_split[8]
		level = line_split[9]
		chrom, pos = locus.split(":")
		chrom = add_chr(chrom)
		pos = int(pos)
		converted = lo.convert_coordinate(chrom, pos)
		if len(converted) > 0:
			for item in converted:
				new_chrom = rem_chr(item[0])
				new_pos = item[1]
				print(f"{new_chrom}\t{new_pos}\t{level}", file = output)

	f.close()
	output.close()

if __name__=="__main__":
	main()

