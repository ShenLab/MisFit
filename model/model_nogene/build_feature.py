import json
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import gzip
from Bio import SeqIO
from constants import log_dir, setting

L = setting['segment_length']
AA_table = setting["AA_table"]
SS_DIM = 7

geneset = pd.read_table(setting["geneset"], sep = "\t")
uniprot_dict = {}
for i, row in geneset.iterrows():
	uniprot_dict[row['UniprotID']] = i

# input positions start from 1, end included
# fixed length embedding for each position, start from 1
def _gen_embedding(folder, uniprot_id, frac, start, end):
	filename = f"{folder}/{uniprot_id}_{frac}.npy"
	data = np.load(os.path.expanduser(filename))
	embedding = np.zeros(shape = (L, setting["ESM_dimension"][setting["ESM"]]))
	embedding[0:(end - start + 1)] = data[start:(end + 1)]
	return embedding.astype(np.float32)

# whole protein embedding through pooling
def _gen_poolembedding(folder, uniprot_id):
	filename = f"{folder}/{uniprot_id}.npy"
	embedding = np.load(os.path.expanduser(filename))
	return embedding.astype(np.float32)

def _gen_logits(folder, uniprot_id, start, end):
	filename = f"{folder}/{uniprot_id}.npy"
	data = np.load(os.path.expanduser(filename))
	logits = np.zeros(shape = (L, len(AA_table)))
	logits[0:(end - start + 1)] = data[(start - 1):end]
	return logits.astype(np.float32)

def _gen_count(folder, uniprot_id, start, end):
	filename = f"{folder}/{uniprot_id}_count.npy"
	data = np.load(os.path.expanduser(filename))
	logits = np.zeros(shape = (L, len(AA_table)))
	logits[0:(end - start + 1)] = data[(start - 1):end]
	return logits.astype(np.float32)

def _gen_contacts(folder, uniprot_id, frac, start, end):
	filename = f"{folder}/{uniprot_id}_{frac}.npy"
	data = np.load(os.path.expanduser(filename))
	contacts = np.zeros(shape = (L, L, 1))
	contacts[0:(end - start + 1), 0:(end - start + 1), 0] = data[(start - 1):end, (start - 1):end]
	return contacts.astype(np.float32)

def _gen_contacts_decomp(folder, uniprot_id, frac, start, end):
	filename = f"{folder}/{uniprot_id}_{frac}.npy"
	data = np.load(os.path.expanduser(filename))
	contacts = np.zeros(shape = (L, L))
	contacts[0:(end - start + 1), 0:(end - start + 1)] = data[(start - 1):end, (start - 1):end]
	return contacts.astype(np.float32)

# coev norm start from 0
def _gen_coev_norm(folder, uniprot_id, frac, start, end):
	filename = f"{folder}/{uniprot_id}_{frac}.npy"
	data = np.load(os.path.expanduser(filename))
	coev = np.zeros(shape = (L, L, 1))
	coev[0:(end - start + 1), 0:(end - start + 1), 0] = data[(start - 1):end, (start - 1):end]
	return coev.astype(np.float32)

def _gen_MSA(folder, uniprot_id, start, end):
	msa_full = np.load(os.path.expanduser(f"{folder}/{uniprot_id}_MSA.npy"))
	full_depth = msa_full.shape[-1]
	sample_index = np.random.randint(low = 0, high = full_depth, size = max_depth)
	sample_index[0] = 0
	msa_full = msa_full[:, sample_index]
	msa = np.ones(shape = (L, max_depth)) * len(AA_table)
	msa[0:(end - start + 1)] = msa_full[(start - 1):end]
	msa = tf.one_hot(msa, len(AA_table)+1, dtype = tf.float32)
	return msa

def _gen_struct(folder, uniprot_id, frac, start, end):
	filename = f"{folder}/coords_{uniprot_id}-F{frac}.gz"
	struc_table = pd.read_table(os.path.expanduser(filename))
	struc_table.set_index("AApos", inplace = True)
	struc_subset = struc_table.loc[start:end]
	struc_feature = np.zeros(shape = (L, SS_DIM))
	ss = struc_subset[["BEND","HELX_LH_PP_P","HELX_RH_3T_P","HELX_RH_AL_P","HELX_RH_PI_P","STRN","TURN_TY1_P"]]
	struc_feature[0:(end-start+1)] = ss.to_numpy()
	N = np.random.rand(L, 3)
	N[0:(end-start+1)] = struc_subset[['N_x', 'N_y', 'N_z']].to_numpy()
	C = np.random.rand(L, 3)
	C[0:(end-start+1)] = struc_subset[['C_x', 'C_y', 'C_z']].to_numpy()
	Ca = np.random.rand(L, 3)
	Ca[0:(end-start+1)] = struc_subset[['CA_x', 'CA_y', 'CA_z']].to_numpy()
	return N, Ca, C, struc_feature

def _gen_refseq(folder, uniprot_id, start, end):
	filename = f"{folder}/{uniprot_id}.fasta.gz"
	with gzip.open(os.path.expanduser(filename), "rt") as f:
		record = SeqIO.read(f, "fasta")
	refseq = [record.seq[i] for i in range((start - 1), end)]
	refseq = [AA_table.get(aa, -1) for aa in refseq]
	refseq_feature = np.zeros(shape = (L, len(AA_table)))
	refseq_feature[0:(end-start+1)] = tf.one_hot(refseq, len(AA_table), dtype = tf.float32)
	return refseq_feature
	
def _gen_overlap_weight(start, end, mask_start, mask_end):
	full_weights = np.ones(shape = (L, 1))
	full_weights[(end - start + 1):, :] = 0.
	if setting['overlap_mask_type'] == "weighted":
		weights = np.array([(i+1)/(setting['overlap_length']+1) for i in range(setting['overlap_length'])])
		weights = np.reshape(weights, (setting['overlap_length'], 1))
		if mask_start > start:
			full_weights[0:setting['overlap_length']] = weights
		if mask_end < end:
			full_weights[(L-setting['overlap_length']):L] = weights[::-1,:]
	else:
		full_weights[:(mask_start - start), :] = 0.
		full_weights[(mask_end - start + 1):, :] = 0.
	return full_weights.astype(np.float32)


def _gen_pop(mu_folder, pop_folder, uniprot_id, start, end):
	mu_full = np.load(os.path.expanduser(f"{mu_folder}/{uniprot_id}_mu.npy"))
	AN_full = np.load(os.path.expanduser(f"{pop_folder}/{uniprot_id}_AN.npy"))
	AC_full = np.load(os.path.expanduser(f"{pop_folder}/{uniprot_id}_AC.npy"))
	AA_mask_full = np.load(os.path.expanduser(f"{pop_folder}/{uniprot_id}_AA_mask.npy"))
	# slice
	mu = np.zeros(shape = (L, len(AA_table)))
	mu[0:(end - start + 1)] = mu_full[(start - 1):end]
	AN = np.zeros(shape = (L, len(AA_table)))
	AN[0:(end - start + 1)] = AN_full[(start - 1):end]
	AC = np.zeros(shape = (L, len(AA_table)))
	AC[0:(end - start + 1)] = AC_full[(start - 1):end]
	AA_mask = np.zeros(shape = (L, len(AA_table)))
	AA_mask[0:(end - start + 1)] = AA_mask_full[(start - 1):end]		
	return mu.astype(np.float32), AN.astype(np.float32), AC.astype(np.float32), AA_mask.astype(np.float32)

def _gen_target(folder, uniprot_id, start, end, split = True):
	target_file = os.path.expanduser(f"{folder}/{uniprot_id}_target.npy")
	target = np.zeros(shape = (L, len(AA_table)))
	train_mask = np.zeros(shape = (L, len(AA_table)))
	val_mask = np.zeros(shape = (L, len(AA_table)))
	if not os.path.exists(target_file):
		return target.astype(np.float32), train_mask.astype(np.float32), val_mask.astype(np.float32)
	target_full = np.load(target_file)
	target[0:(end - start + 1)] = target_full[(start - 1):end]
	if split:
		mask_full = np.load(os.path.expanduser(f"{folder}/{uniprot_id}_train_mask.npy"))
		train_mask[0:(end - start + 1)] = mask_full[(start - 1):end]
	mask_full = np.load(os.path.expanduser(f"{folder}/{uniprot_id}_val_mask.npy"))
	val_mask[0:(end - start + 1)] = mask_full[(start - 1):end]
	return target.astype(np.float32), train_mask.astype(np.float32), val_mask.astype(np.float32)

def _gen_features(df_list, shuffle = True, pop = True, target_folder = None, split = True):
	if shuffle:
		df_list = df_list.sample(frac = 1)
	for _, row in df_list.iterrows():
		uniprot_id = row['UniprotID']
		frac = row['frac']
		start = row['start']
		end = row['end']
		struc_frac = row['struc_frac']
		struc_start = row['struc_start']
		struc_end = row['struc_end']
		mask_start = row['unmask_start']
		mask_end = row['unmask_end']
		l = end - start + 1
		weights = _gen_overlap_weight(start, end, mask_start, mask_end)
		inputs = {"id": uniprot_dict[uniprot_id], "uniprot": uniprot_id, "frac": frac, "l": l, "overlap_weight": weights}
		if "ESM" in setting["feature"]:
			embedding = _gen_embedding(f"{setting['ESM_folder'][setting['ESM']]}/repr_{setting['segment_length']}_{setting['overlap_length']}/", uniprot_id, frac, 1, end - start + 1)
			inputs["embedding"] = embedding
		if "pooling" in setting['feature']:
			poolembedding = _gen_poolembedding(f"{setting['ESM_folder'][setting['ESM']]}/{setting['pooling_method']}/", uniprot_id)
			inputs['poolembedding'] = poolembedding
		if "logits" in setting['feature']:
			logits = _gen_logits(f"{setting['ESM_folder'][setting['ESM']]}/logits_np/", uniprot_id, start, end)
			inputs['logits'] = logits
		if "coev_norm" in setting["feature"]:
			coev = _gen_coev_norm(f"{setting['MSA_folder']}/coev_norm_{setting['segment_length']}_{setting['overlap_length']}/", uniprot_id, frac, 1, end - start + 1)
			inputs["coev"] = coev
		if "contact" in setting["feature"]:
			contact = _gen_contacts(f"{setting['ESM_folder'][setting['ESM']]}/contact_{setting['segment_length']}_{setting['overlap_length']}/", uniprot_id, frac, 1, end - start + 1)
			inputs["contact"] = contact
		if "contact_decomp" in setting["feature"]:
			contact_decomp = _gen_contacts_decomp(f"{setting['ESM_folder'][setting['ESM']]}/contact_decomp_{setting['segment_length']}_{setting['overlap_length']}_{setting['cov_weight']}/", uniprot_id, frac, 1, end - start + 1)
			inputs["contact_decomp"] = contact_decomp
		if "MSA" in setting["feature"]:
			msa = _gen_MSA(f"{setting['MSA_folder']}/", uniprot_id, start, end, max_depth = setting['MSA_depth'])
			inputs["MSA"] = msa
#			inputs["cover"] = cover
		if "MSA_count" in setting["feature"]:
			MSA_count = _gen_count(f"{setting['MSA_count_folder']}/", uniprot_id, start, end)
			inputs["MSA_count"] = MSA_count
		if "struct" in setting["feature"]:
			N, Ca, C, struc_feature = _gen_struct(setting["struct_folder"], uniprot_id, struc_frac, struc_start, struc_end)
			inputs['N_coord'] = N
			inputs['Ca_coord'] = Ca
			inputs['C_coord'] = C
			inputs['struct'] = struc_feature
		if "refseq" in setting["feature"]:
			refseq_feature = _gen_refseq(setting["refseq_folder"], uniprot_id, start, end)
			inputs['refseq'] = refseq_feature
		if pop:
			mu, AN, AC, AA_mask = _gen_pop(setting["mu_folder"], setting["pop_folder"], uniprot_id, start, end)
			AC = np.clip(AC, 0, AN * setting['max_AF'])
			mu = mu * setting["mu_scaler"]
#			AA_mask = AA_mask * weights
			inputs.update({"mu": mu, "AN": AN, "AA_mask": AA_mask, "AC": AC})
		if target_folder is not None:
			target, train_mask, val_mask = _gen_target(target_folder, uniprot_id, start, end, split = split)
			if split:
				train_mask = train_mask * weights
				inputs['train_mask'] = train_mask
			val_mask = val_mask * weights
			inputs['val_mask'] = val_mask
			inputs['target'] = target
		yield inputs

def _gen_signature(pop = True, target_folder = None, split = True):
	input_signature = {
	                   "id": tf.TensorSpec(shape=(), dtype = tf.int32),
	                   "uniprot": tf.TensorSpec(shape=(), dtype = tf.string),
	                   "frac": tf.TensorSpec(shape=(), dtype=tf.int32),
	                   "l": tf.TensorSpec(shape=(), dtype=tf.int32),
	                   "overlap_weight": tf.TensorSpec(shape=(L, 1), dtype=tf.float32),
	                  }
	if "ESM" in setting["feature"]:
		input_signature["embedding"] = tf.TensorSpec(shape=(L, setting["ESM_dimension"][setting["ESM"]]), dtype=tf.float32)
	if "pooling" in setting['feature']:
		input_signature['poolembedding'] = tf.TensorSpec(shape=(setting["ESM_dimension"][setting["ESM"]]), dtype = tf.float32)
	if "logits" in setting['feature']:
		input_signature['logits'] = tf.TensorSpec(shape = (L, len(AA_table)), dtype = tf.float32)
	if "coev_norm" in setting["feature"]:
		input_signature["coev"] = tf.TensorSpec(shape=(L, L, 1), dtype=tf.float32)
	if "contact" in setting["feature"]:
		input_signature["contact"] = tf.TensorSpec(shape=(L, L, 1), dtype = tf.float32)
	if "contact_decomp" in setting["feature"]:
		input_signature["contact_decomp"] = tf.TensorSpec(shape=(L, L), dtype = tf.float32)
	if "MSA" in setting["feature"]:
		input_signature["MSA"] = tf.TensorSpec(shape=(L, setting['MSA_depth'], len(AA_table)+1), dtype=tf.float32)
#		input_signature["cover"] = tf.TensorSpec(shape=(setting['MSA_depth']), dtype=tf.float32)
	if "MSA_count" in setting['feature']:
		input_signature['MSA_count'] = tf.TensorSpec(shape = (L, len(AA_table)), dtype = tf.float32)
	if "struct" in setting["feature"]:
		input_signature["N_coord"] = tf.TensorSpec(shape=(L, 3), dtype=tf.float32)
		input_signature["Ca_coord"] = tf.TensorSpec(shape=(L, 3), dtype=tf.float32)
		input_signature["C_coord"] = tf.TensorSpec(shape=(L, 3), dtype=tf.float32)
		input_signature["struct"] = tf.TensorSpec(shape=(L, SS_DIM), dtype=tf.float32)
	if "refseq" in setting["feature"]:
		input_signature["refseq"] = tf.TensorSpec(shape=(L, len(AA_table)), dtype = tf.float32)
	if pop:
		input_signature.update({
		                        "mu": tf.TensorSpec(shape=(L, len(AA_table)), dtype=tf.float32),
		                        "AN": tf.TensorSpec(shape=(L, len(AA_table)), dtype=tf.float32),
		                        "AA_mask": tf.TensorSpec(shape=(L, len(AA_table)), dtype=tf.float32),
		                       })
		input_signature.update({"AC": tf.TensorSpec(shape=(L, len(AA_table)), dtype=tf.float32)})
	if target_folder is not None:
		if split:
			input_signature['train_mask'] = tf.TensorSpec(shape=(L, len(AA_table)), dtype=tf.float32)
		input_signature['val_mask'] = tf.TensorSpec(shape=(L, len(AA_table)), dtype=tf.float32)
		input_signature['target'] = tf.TensorSpec(shape=(L, len(AA_table)), dtype=tf.float32)
	return input_signature

def gen_data(list_ids, shuffle = True, pop = True, target = None, split = True, batch_size = setting['batch_size'], repeat = -1):
	if not (isinstance(list_ids, list)):
		list_ids = [list_ids]
	df = pd.DataFrame()
	for list_id in list_ids:
		df = pd.concat([df, pd.read_table(os.path.expanduser(f"{setting['data_folder']}/list_{list_id}.txt"))])
	if len(list_ids) > 1:
		df = df.drop_duplicates()
	signatures = _gen_signature(pop, target, split)
	data = tf.data.Dataset.from_generator(
	                                      lambda: _gen_features(df, shuffle, pop, target, split),
	                                      output_signature = signatures
	                                     ).repeat(repeat)
	data = data.batch(batch_size)
	data = data.prefetch(4)
	return data

def get_data_size(list_ids):
	if not (isinstance(list_ids, list)):
		list_ids = [list_ids]
	df = pd.DataFrame()
	for list_id in list_ids:
		df = pd.concat([df, pd.read_table(os.path.expanduser(f"{setting['data_folder']}/list_{list_id}.txt"))])
	if len(list_ids) > 1:
		df = df.drop_duplicates()
	return len(df)

