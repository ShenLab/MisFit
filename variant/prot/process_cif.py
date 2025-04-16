import pdbecif.mmcif_io as mmcif
import pandas as pd
import os
import re
import gzip

def process_atom(atom_df, atom):
	selected_df = atom_df.loc[atom_df['label_atom_id'] == atom, ['label_seq_id', 'Cartn_x', 'Cartn_y', 'Cartn_z']]
	selected_df.columns = ['AApos', atom + '_x', atom + '_y', atom + '_z']
	selected_df['AApos'] = selected_df['AApos'].astype('int')
	return selected_df

def process_cif(prot_label, return_struc = True):
	# read cif to dict
	keyname = 'AF-' + prot_label
	filename = 'alphafold2_files/' + keyname + '-model_v2.cif.gz'
	cfr = mmcif.CifFileReader(input = 'data')
	cif_dict = cfr.read(filename, output = 'cif_dictionary')
	cif_dict = cif_dict[keyname]
	if return_struc and ('_struct_conf' in cif_dict):
		struct_dict = {key:cif_dict['_struct_conf'][key] for key in ['beg_label_seq_id', 'end_label_seq_id', 'conf_type_id']}
		for key in struct_dict:
			if not isinstance(struct_dict[key], list):
				struct_dict[key] = [struct_dict[key]]
		struct_df = pd.DataFrame(struct_dict)
		struct_df.columns = ['beg', 'end', 'type']
		struct_df['beg'] = struct_df['beg'].astype('int')
		struct_df['end'] = struct_df['end'].astype('int')
	else:
		struct_df = None
	atom_df = pd.DataFrame(cif_dict['_atom_site'])
	seq = cif_dict['_struct_ref']['pdbx_seq_one_letter_code'].replace('\n','')
	# reference AA
	info_df = pd.DataFrame({'AApos': [i+1 for i in range(len(seq))], 'refAA': [i for i in seq]})
	# 3d coordinates of N, C, Ca
	info_df = info_df.merge(process_atom(atom_df, 'N'), how = 'left', on = 'AApos')
	info_df = info_df.merge(process_atom(atom_df, 'C'), how = 'left', on = 'AApos')
	info_df = info_df.merge(process_atom(atom_df, 'CA'), how = 'left', on = 'AApos')
	info_df.set_index('AApos', inplace = True)
	info_df.sort_index(inplace = True)
	info_df = info_df.sort_values(by = 'AApos')
	return info_df, struct_df

def add_struct(info_df, struct_df, struct_list):
	for struct in struct_list:
		info_df[struct] = 0
	if struct_df is None:
		return info_df
	for index, row in struct_df.iterrows():
		if row['type'] in struct_list:
			info_df.loc[row['beg']:row['end'], row['type']] = 1
	return info_df

def main():
	geneset = pd.read_table("geneset_uniprot.txt")
	geneset['Parts'] = 0
	all_struct = []
	with open("struct_labels.txt", "r") as f:
		for line in f:
			all_struct.append(line.strip())
	
	for index, row in geneset.iterrows():
		gene = row['UniprotID']
		part = 1
		prot_label = gene + "-F" + str(part)
		while os.path.isfile("alphafold2_files/AF-" + prot_label + "-model_v2.cif.gz"):
			geneset.loc[index, 'Parts'] = part
			info_df, struct_df = process_cif(prot_label)
			info_df = add_struct(info_df, struct_df, all_struct)
			info_df = info_df.reset_index()
			info_df.to_csv("AF2_table/coords_" + prot_label + ".gz", sep = "\t", index = False)
			part += 1
			prot_label = gene + "-F" + str(part)

	geneset.to_csv("geneset_uniprot_parts.txt", sep = "\t", index = False)

if __name__=="__main__":
	main()



