import pandas as pd
from os.path import exists
import json
def main():
	geneset = pd.read_table("../../list/geneset_uniprot.txt")
	for _, row in geneset.iterrows():
		uniprot_id = row['UniprotID']
		transcript_id = row['TranscriptID']
		inputname = f"var_by_uniprot/{uniprot_id}_info.txt"
		if exists(inputname):
			var_df = pd.read_table(inputname)
			var_df = var_df.drop_duplicates()
			function_df = pd.read_table(f"../../functional/scores/{transcript_id}_scores.txt.gz")
			function_df = pd.merge(var_df, function_df)
			mapping_file = f"../../pep/ensembl_to_uniprot_pos/{transcript_id}.json"
			if exists(mapping_file):
				with open(mapping_file, "r") as f:
					mapping = json.load(f)['mapping']
				function_df['Uniprot_position'] = function_df['Protein_position'].astype(str).map(mapping)
			else:
				function_df['Uniprot_position'] = function_df['Protein_position']
			function_df = function_df.dropna(subset = ['Uniprot_position'])
			function_df['Uniprot_position'] = function_df['Uniprot_position'].astype('int')
			function_df.to_csv(f"var_by_uniprot_function/{uniprot_id}.txt.gz", index = False, sep = "\t")
			
		

if __name__=="__main__":
	main()
