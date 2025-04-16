import os
import re
import gzip
import pdbecif.mmcif_io as mmcif

def main():
	files_list = os.listdir("alphafold2_files/")
	all_struct = []
	for filename in files_list:
		keyname_search = re.search(r"(.+)-model_v2.cif.gz", filename)
		if keyname_search:
			cfr = mmcif.CifFileReader(input = 'data')
			cif_dict = cfr.read("alphafold2_files/" + filename, output = 'cif_dictionary')
			keyname = keyname_search.group(1)
			cif_dict = cif_dict[keyname]
			if '_struct_conf_type' in cif_dict:
				structs = cif_dict['_struct_conf_type']['id']
				if isinstance(structs, list):
					for struct in structs:
						if struct in all_struct:
							continue
						print(keyname, struct)
						all_struct.append(struct)
				elif isinstance(structs, str):
					if structs in all_struct:
						continue
					print(keyname, structs)
					all_struct.append(structs)
			else:
				print(keyname, "undefined")
	output = open("struct_labels.txt", "w")
	for struct in all_struct:
		print(struct, file = output)
	output.close()

if __name__=="__main__":
	main()

