import pandas as pd
import pyranges as pr

def main():
	original = pd.read_table("all_pos.txt", names = ['Chromosome', 'Start', 'End'])
	original['End'] = original['End'] + 1
	added = pd.read_table("add_pos.txt", names = ['Chromosome', 'Start', 'End'])
	added['End'] = added['End'] + 1
	original_pr = pr.PyRanges(original)
	added_pr = pr.PyRanges(added)
	diff_pr = added_pr.subtract(original_pr)
	diff_pr = diff_pr.sort()
	diff_pr = diff_pr.merge()
	diff_pr.End = diff_pr.End - 1
	diff_pr.to_csv("diff_pos.txt", sep = "\t", header = False)

if __name__="__main__":
	main()

