import json
from os.path import exists

def remap(transcript_id, AApos, mapping):
	if str(AApos) in mapping['mapping']:
		return str(mapping['mapping'][str(AApos)])
	else:
		return None

def main():
	id_dict = {}
	with open("../../list/geneset_uniprot.txt", "r") as f:
		line = f.readline()
		for line in f:
			line_split = line.split("\t")
			id_dict[line_split[2]] = line_split[0]
	num_pos = 0
	num_neg = 0
	
#	with open("/share/terra/Projects/MVP/gMVP/genetic_variants_data/Cancer_Hotspot/Cancer_Hotspot.csv", "r") as f:
	with open("/share/terra/Projects/MVP/gMVP/genetic_variants_data/Cancer_Hotspot/DiscovEHR_all_genes_v1.csv", "r") as f:
		line = f.readline()
		for line in f:
			line_split = line.strip().split("\t")
			transcript_id = line_split[6]
			if transcript_id in id_dict:
				uniprot_id = id_dict[transcript_id]
			else:
				continue
			AApos = line_split[10]
			AAref = line_split[7]
			AAalt = line_split[8]
			target = line_split[15]
			map_file = "../../pep/ensembl_to_uniprot_pos/" + transcript_id + ".json"
			if exists(map_file):
				with open(map_file, "r") as d:
					mapping = json.load(d)
				new_pos = remap(transcript_id, AApos, mapping)
				if new_pos is None:
					continue
			else:
				new_pos = AApos
			with open("label_by_uniprot/" + uniprot_id + ".txt", "a") as outfile:
				print(new_pos, AAref, AAalt, target, sep = "\t", file = outfile)
			if target == "1":
				num_pos += 1
			elif target == "0":
				num_neg += 1

	print(f"pos: {num_pos}, neg: {num_neg}")

			

if __name__=="__main__":
	main()
