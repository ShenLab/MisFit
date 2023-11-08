import json
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import gzip
from Bio import SeqIO
import random

with open("setting.json", "r") as setting_file:
	setting = json.load(setting_file)

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices([gpus[i] for i in setting["GPU"]], 'GPU')

geneset = pd.read_table(setting["geneset"], sep = "\t")
uniprot_dict = {}
for i, row in geneset.iterrows():
	uniprot_dict[row['UniprotID']] = i

def _gen_features(pop_folder, geneset, shuffle = True, batch_size = setting['batch_size']):
	if shuffle:
		geneset = geneset.sample(frac = 1)
	for _, row in geneset.iterrows():
		uniprot_id = row['UniprotID']
		filename = os.path.expanduser(f"{pop_folder}/{uniprot_id}_info.txt.gz")
		if not os.path.exists(filename):
			continue
		df = pd.read_table(filename)
		uniprot_index = uniprot_dict[uniprot_id]
		AN = df[['AN']].to_numpy(dtype = np.float32)
		AC = df[['AC']].to_numpy(dtype = np.float32)
		AC = np.clip(AC, 0., setting['max_AF'] * AN)
		mu = df[['mu']].to_numpy(dtype = np.float32)
		mask = np.ones_like(AC, dtype = np.float32)
		total_nvar = mask.sum()
		for i in range(0, int(total_nvar), setting['max_var']):
			l = AN[i:(i + setting['max_var'])].shape[0]
			inputs = {"id": uniprot_dict[uniprot_id]} # ()
			inputs['AN'] = np.pad(AN[i:(i + l)], ((0, setting['max_var'] - l), (0, 0))) # (max_var, 1)
			inputs['AC'] = np.pad(AC[i:(i + l)], ((0, setting['max_var'] - l), (0, 0))) # (max_var, 1)
			inputs['mu'] = np.pad(mu[i:(i + l)], ((0, setting['max_var'] - l), (0, 0))) # (max_var, 1)
			inputs['mask'] = np.pad(mask[i:(i + l)], ((0, setting['max_var'] - l), (0, 0))) # (max_var, 1)
			weight = mask[i:(i + l)].sum() / total_nvar
			inputs['weight'] = weight # ()
			yield inputs

def gen_data(pop_folder = setting['pop_folder'], geneset = geneset, shuffle = True, batch_size = setting['batch_size']):
	input_signatures = {
	                    "id": tf.TensorSpec(shape=(None), dtype = tf.int64), 
	                    "mu": tf.TensorSpec(shape=(None, 1), dtype = tf.float32), 
	                    "AN": tf.TensorSpec(shape=(None, 1), dtype = tf.float32),
						"AC": tf.TensorSpec(shape=(None, 1), dtype = tf.float32),
						"mask": tf.TensorSpec(shape=(None, 1), dtype = tf.float32),
						"weight": tf.TensorSpec(shape=(None), dtype = tf.float32)
	                   }
	data = tf.data.Dataset.from_generator(
	                                      lambda: _gen_features(pop_folder, geneset, shuffle),
	                                      output_signature = input_signatures,
	                                     )
	data = data.batch(batch_size)
	return data


