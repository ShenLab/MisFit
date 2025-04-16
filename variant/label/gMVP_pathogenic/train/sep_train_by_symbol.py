import json
from os.path import exists
import random

ratio = 0.9
random.seed(0)


def main():
	
	with open("train.csv", "r") as f:
		line = f.readline()
		for line in f:
			line_split = line.strip().split("\t")
			symbol = line_split[5]
			
			chrom = line_split[1]
			pos = line_split[2]
			ref = line_split[3]
			alt = line_split[4]
			target = line_split[15]
			if random.random() < ratio:
				split = 1
			else:
				split = 0
			with open("var_by_symbol/" + symbol + ".txt", "a") as outfile:
				print(chrom, pos, ref, alt, target, split, sep = "\t", file = outfile)

			

if __name__=="__main__":
	main()
