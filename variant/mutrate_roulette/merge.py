import gzip
import re

def main():
	files = []
	with open("roulette_vcfs_ordered.txt", "r") as f:
		for line in f:
			files.append(line.strip())
	output = open("roulette_mu.vcf", "a")
	for filename in files:
		f = gzip.open(filename, "rt")
		for line in f:
			if re.search(r"^#", line):
				continue
			print(line, end = "", file = output)
		f.close()

	output.close()

if __name__ == "__main__":
	main()
