import pandas as pd
from os.path import exists
import json

def main():
	geneset = pd.read_table("../list/geneset_uniprot.txt")
	for _, row in geneset.iterrows():
		uniprot_id = row['UniprotID']
		transcript_id = row['TranscriptID']
		gene_id = row['GeneID']
		symbol = row['Symbol']
		function_df = pd.read_table(f"scores/{transcript_id}_scores.txt.gz")
		mapping_file = f"../pep/ensembl_to_uniprot_pos/{transcript_id}.json"
		if exists(mapping_file):
			with open(mapping_file, "r") as f:
				mapping = json.load(f)['mapping']
			function_df['Uniprot_position'] = function_df['Protein_position'].astype(str).map(mapping)
		else:
			function_df['Uniprot_position'] = function_df['Protein_position']
#		function_df['Uniprot_position'] = function_df['Uniprot_position'].astype('int')
		function_df['UniprotID'] = uniprot_id
		function_df['GeneID'] = gene_id
		function_df['Symbol'] = symbol
		function_df.to_csv(f"scores2/{uniprot_id}_scores.txt.gz", index = False, sep = "\t")

if __name__=="__main__":
	main()
