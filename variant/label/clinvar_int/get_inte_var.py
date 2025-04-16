import pandas as pd
from os.path import exists

def main():
	geneset = pd.read_table("../../list/geneset_uniprot_len.txt")
	for _, row in geneset.iterrows():
		uniprot_id = row['UniprotID']
		symbol = row['Symbol']
		varfile1 = f"../clinvar_clean/var_by_uniprot/{uniprot_id}_info.txt"
		varfile2 = f"../PrimateAI3D_benign/mis_benign/{uniprot_id}.txt.gz"
		varfile0 = f"../PrimateAI3D_benign/snv_only/{uniprot_id}.txt.gz"
		if exists(varfile1) and exists(varfile0):
			df1 = pd.read_table(varfile1)
			df0 = pd.read_table(varfile0)
			df1 = pd.merge(df0, df1)
			df1['Source'] = "Clinvar"
		else:
			continue
		if exists(varfile2):
			df2 = pd.read_table(varfile2)
			df2['Source'] = "PrimateAI3D"
			df2['Recent'] = pd.NA
			df2['Label'] = 0
			df2['Symbol'] = symbol
			df2['UniprotID'] = uniprot_id
		else:
			df2 = pd.DataFrame()
		df = pd.concat([df1, df2], ignore_index = True)
		df = df.drop_duplicates(subset = ['Pos', 'Alt', 'Label'])
		df = df.drop_duplicates(subset = ['Pos', 'Alt'], keep = False)
		var1_damage = df[(df['Source']=="Clinvar") & (df['Label']==1)]
		var1_benign = df[(df['Source']=="Clinvar") & (df['Label']==0)]
		var2_benign = df[(df['Source']=="PrimateAI3D") & (df['Label']==0)]
		if len(var1_damage) > len(var1_benign):
			if len(var1_damage) > len(var1_benign) + len(var2_benign):
				var1_damage = var1_damage.sample(n = len(var1_benign) + len(var2_benign))
#				pass
			else:
				var2_benign = var2_benign.sample(n = len(var1_damage) - len(var1_benign))
			df_select = pd.concat([var1_damage, var1_benign, var2_benign])
		else:
			var1_benign = var1_benign.sample(n = len(var1_damage))
			df_select = pd.concat([var1_damage, var1_benign])
		if len(df_select) > 0:
			df_select.to_csv(f"var_by_uniprot/{uniprot_id}.txt.gz", index = False, sep = "\t")

if __name__=="__main__":
	main()

