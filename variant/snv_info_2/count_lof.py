import gzip
import re

def main():
	filename = "all_pos_alt_HClof_sep_canonical_AF.vcf.gz"
	stop_gained = 0
	other = 0
	f = gzip.open(filename, "rt")
	for line in f:
		if re.search(r"^#", line):
			continue
		else:
			if re.search(r"stop_gained", line):
				stop_gained += 1
			else:
				other += 1
	f.close()
	print(stop_gained, other)

if __name__=="__main__":
	main()
