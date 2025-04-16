import sys
import re
import gzip

def main(filename):
	f = open(filename, "r")
	for line in f:
		if re.search(r"^#CHROM", line):
			fields = line.strip().split("\t")
			fields.pop()
			print(*fields, sep = "\t", end = "\n")
			break
		else:
			print(line, end = "")
	for line in f:
		print(line, end = "")
	f.close()

if __name__=="__main__":
	filename = sys.argv[1]
	main(filename)

