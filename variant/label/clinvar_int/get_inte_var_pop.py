import pandas as pd
from os.path import exists

def main():
	geneset = pd.read_table("../../list/geneset_uniprot_len.txt")
	for _, row in geneset.iterrows():
		uniprot_id = row['UniprotID']
		symbol = row['Symbol']
		varfile1 = f"../clinvar_clean/var_by_uniprot/{uniprot_id}_info.txt"
		varfile0 = f"../../snv_info_2/mis_info_by_protein_mapping/{uniprot_id}_info.txt.gz"
		if exists(varfile1) and exists(varfile0):
			df1 = pd.read_table(varfile1)
			df1 = df1[df1['Label'] == 1]
			df0 = pd.read_table(varfile0)
			df0['Uniprot_position'] = df0['Uniprot_AA_pos']
			df0 = df0[df0['Uniprot_position'].notna()]
			df1 = pd.merge(df0[['Pos', 'Alt', 'Uniprot_position', 'AA_ref', 'AA_alt']], df1)
			df1['Source'] = "Clinvar"
			df0 = df0.loc[(df0['roulette_mu']<1e-7)&(df0['UKBB_AC']/df0['UKBB_AN']>1e-5), ['Chrom', 'Pos', 'Ref', 'Alt', 'Uniprot_position', 'AA_ref', 'AA_alt']]
			df0['Source'] = "UKBB"
			df0['Recent'] = pd.NA
			df0['Label'] = 0
			df0['Symbol'] = symbol
			df0['UniprotID'] = uniprot_id
		df = pd.concat([df0, df1], ignore_index = True)
		df = df.drop_duplicates(subset = ['Pos', 'Alt', 'Label'])
		df = df.drop_duplicates(subset = ['Pos', 'Alt'], keep = False)
		var_damage = df[(df['Source']=="Clinvar") & (df['Label']==1)]
		var_benign = df[(df['Source']=="UKBB") & (df['Label']==0)]
		if len(var_damage) > len(var_benign):
			var_damage = var_damage.sample(n = len(var_benign))
		else:
			var_benign = var_benign.sample(n = len(var_damage))
		df_select = pd.concat([var_damage, var_benign])
		if len(df_select) > 0:
			df_select.to_csv(f"var_by_uniprot_pop/{uniprot_id}.txt.gz", index = False, sep = "\t")

if __name__=="__main__":
	main()

