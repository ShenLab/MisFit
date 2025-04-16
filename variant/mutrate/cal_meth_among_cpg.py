import gzip
import re

def main():
	meth_count = [0,0,0]
	f = gzip.open("all_coding_alt_gnomad_mu.vcf.gz", "rt")
	for line in f:
		if re.search(r"^#", line):
			continue
		info = line.strip().split("\t")[7]
		if re.search("CpG", info):
			if re.search("METH=2", info):
				meth_count[2] += 1
			elif re.search("METH=1", info):
				meth_count[1] += 1
			else:
				meth_count[0] += 1
	meth_rate = [count / sum(meth_count) for count in meth_count]
	print(*meth_rate)
	f.close()

if __name__=="__main__":
	main()
