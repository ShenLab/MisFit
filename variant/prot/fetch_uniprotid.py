import os
import re

def main():
	files_list = os.listdir("alphafold2_files/")
	all_ids = set()
	for filename in files_list:
		uniprotid_search = re.search(r"AF-(\w+)-F1-model_v2.pdb.gz", filename)
		if uniprotid_search:
			uniprotid = uniprotid_search.group(1)
			all_ids.add(uniprotid)
	output = open("alphafold2_uniprotid.txt", "w")
	for uniprotid in all_ids:
		print(uniprotid, file = output)

if __name__=="__main__":
	main()

