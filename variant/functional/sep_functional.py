import gzip
import re
import json
from os.path import exists

def main():
	input_filename = "all_pos_alt_missense_sep_canonical_functional.vcf.gz"
	output_dir = "./scores/"
	csq_names_all = ["Allele","Consequence","IMPACT","SYMBOL","Gene","Feature_type","Feature","BIOTYPE","EXON","INTRON","HGVSc","HGVSp","cDNA_position","CDS_position","Protein_position","Amino_acids","Codons","Existing_variation","DISTANCE","STRAND","FLAGS","SYMBOL_SOURCE","HGNC_ID"]
	csq_names = ["Feature", "Protein_position", "Amino_acids"]
	assert csq_names[0] == "Feature"
	csq_indices = [csq_names_all.index(i) for i in csq_names]
	csq_renames = ["TranscriptID", "Protein_position", "Protein_length", "AA_ref", "AA_alt"]
	scores_names = ["CADD_phred", "CADD_raw_rank", "MPC", "MPC_rank", "MVP", "MVP_rank", "M_CAP", "M_CAP_rank", "PrimateAI", "PrimateAI_rank", "REVEL", "REVEL_rank", "EVE", "gMVP", "gMVP_rank"]

	transcripts = set()
	with open("../list/geneset_uniprot.txt", "r") as f:
		line = f.readline()
		line_split = line.strip().split("\t")
		id_index = line_split.index("TranscriptID")
		for line in f:
			line_split = line.strip().split("\t")
			transcript_id = line_split[id_index]
			transcripts.add(transcript_id)
			with open(output_dir + transcript_id + "_scores.txt", "w") as output_file:
				print("Chrom", "Pos", "Ref", "Alt", *csq_renames, *scores_names, sep = "\t", file = output_file)

	input_file = gzip.open(input_filename, "rt")
	count = 0
	for line in input_file:
		if re.search(r"^#", line):
			continue
		count += 1
		if count % 100000 == 0:
			print(f"{count} lines processed")
		line_split = line.strip().split("\t")
		info = line_split[7]
		csq_search = re.search(f"CSQ=([^;]+)", info)
		csq = csq_search.group(1)
		csq_split = csq.split("|")
		transcript_id = csq_split[csq_indices[0]]
		if not (transcript_id in transcripts):
			continue
		fields = [line_split[i] for i in [0,1,3,4]]
		for i in csq_indices:
			fields += csq_split[i].split("/")
		for score_name in scores_names:
			score_search = re.search(r"[;^]%s=([^;]+)" % score_name, info)
			if score_search:
				score = score_search.group(1)
			else:
				score = ""
			fields.append(score)
		with open(output_dir + transcript_id + "_scores.txt", "a") as output_file:
			print(*fields, sep = "\t", file = output_file)
	input_file.close()

if __name__=="__main__":
	main()
