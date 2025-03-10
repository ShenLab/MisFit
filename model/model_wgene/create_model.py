import json
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa
tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers
import sys
from popgen_PIG_SD import PopACModel
from layers import SymAttention, FullPotts, IPABlock, TransformerBlock, RigidFrom3Points, Dense, AddGaussianNoise
from build_feature import gen_data, get_data_size
from os.path import exists
from constants import log_dir, setting


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in setting['GPU']])
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mirrored_strategy = tf.distribute.MirroredStrategy()
L = setting['segment_length']
AA_table = setting['AA_table']

sys.stdout = open(f'{log_dir}/model.log', 'a')

class PriorPosition(tfk.Model):
	def __init__(self, use_cov = False, aa_base = False, **kwargs):
		super().__init__(**kwargs)
		self.use_cov = use_cov
		self.aa_base = aa_base
	def call(self, inputs): # inputs: (B, L, L) decomposed correlation matrix
		shape = tf.shape(inputs)
		mean = tf.zeros([shape[0], shape[1], 1]) # (B, L, 1)
		return mean
	def sample(self, inputs, nsample): # if use cov, input decomposed covariant matrix, else input anything with shape (B, L, A)
		shape = tf.shape(inputs)
		if self.use_cov:
			position_sample_hidden = tfd.Normal(0., 1.).sample([nsample, shape[0], shape[1], 1]) # (nsample, B, l, 1)
			position_sample = tf.einsum("...Ll,...ld->...Ld", inputs, position_sample_hidden) # (nsample, B, L, 1)
			return position_sample
		elif not self.aa_base:
			position_sample = tfd.Normal(0., 1.).sample([nsample, shape[0], shape[1], 1])
			return position_sample # (nsample, B, L, 1)
		else:
			aa_sample = tfd.Normal(0., 1.).sample([nsample, shape[0], shape[1], shape[-1]])
			return aa_sample

class Gene(tfkl.Layer):
	def __init__(self, ngene, smin, smax_mean_global, smax_sd_global, init_smax_mean, **kwargs):
		super().__init__(**kwargs)
		self.ngene = ngene
		self.smin = smin
		self.smax_mean_global = smax_mean_global
		self.smax_sd_global = smax_sd_global
		init_smax_mean = tf.constant(init_smax_mean, shape = (ngene, 1), dtype = tf.float32)
		self.smax_mean_gene = tf.Variable(init_smax_mean, name = "smax_mean_gene", dtype = tf.float32, constraint = lambda z: tfp.math.clip_by_value_preserve_gradient(z, 1e-3 + smin, 0.))
	def call(self, belong_mt): # (B, ngene)
		smax_mean = belong_mt @ self.smax_mean_gene
		return smax_mean # (B, 1)
	def sample(self, belong_mt):
		smax_mean = self.call(belong_mt) # (B, 1)
		smax_global_distr = tfd.Normal(self.smax_mean_global, self.smax_sd_global) # ()
		psmax = smax_global_distr.log_prob(smax_mean) # (B, 1)
		return smax_mean, psmax


class Selection(tfkl.Layer):
	def __init__(self, smin = -13.8, **kwargs):
		super().__init__(**kwargs)
		self.smin = smin
	def call(self, inputs):
		smax, degree = inputs # (B, 1, 1), (nsample, B, L, A)
		s = degree * (smax - self.smin) + self.smin # ( nsample, B, L, A) or (B, L, A)
		return(s)


class ProbModel(tfk.Model):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		geneset = pd.read_table(setting["geneset"], sep = "\t")
		self.total_length = tf.constant(geneset[['Length']], dtype = tf.float32) # (ngene, 1)
		self.gene = Gene(
			ngene = setting['ngene'], 
			smin = setting['min_log_s'], 
			smax_mean_global = setting['smax_mean_global'],
			smax_sd_global = setting['smax_sd_global'],
			init_smax_mean = np.load(os.path.expanduser(setting['init_smax_mean'])), 
		)
		self.prior_position = PriorPosition(use_cov = setting['use_cov'], aa_base = setting['aa_base_sampling'])
		self.selection = Selection(smin = setting['min_log_s'])
		self.popmodel = PopACModel()
		self.d_mean_marginal = setting['d_mean_marginal']
		self.d_sd_marginal = setting['d_sd_marginal']
		self.d_sd = tf.Variable(1., name = "d_sd", dtype = tf.float32, constraint = lambda z: tfp.math.clip_by_value_preserve_gradient(z, 1e-3, setting['d_sd_max']))
		self.ac_loss_tracker = tfk.metrics.Mean(name="ac_loss")
		self.reg_loss_tracker = tfk.metrics.Mean(name="reg_loss")
		self.marginal_loss_tracker = tfk.metrics.Mean(name="marginal_loss")
		self.total_loss_tracker = tfk.metrics.Mean(name="total_loss")
	def call(self, inputs, training = None):
		belong_mt = tf.one_hot(inputs['id'], depth = self.gene.ngene)
		d_trans = tf.broadcast_to(self.d_mean_marginal, tf.shape(inputs['mu']))
		d_sd = tf.broadcast_to(self.d_sd_marginal, tf.shape(inputs['mu']))
		smax_mean = self.gene.call(belong_mt) # (B, 1)
		smax_mean = tf.expand_dims(smax_mean, -1) # (B, 1, 1)
		s = self.selection((smax_mean, tf.math.sigmoid(d_trans))) # (B, L, A)
		return d_trans, s, d_sd, smax_mean # (B, L, A)
	def get_loss(self, inputs, training = None, msample = setting['msample'], nsample = setting['nsample'], rsample = setting['rsample']):
		belong_mt = tf.one_hot(inputs['id'], depth = self.gene.ngene)
		if self.prior_position.use_cov:
			position_sample = self.prior_position.sample(inputs['contact_decomp'], nsample = nsample) # (n, B, L, 1)
		else:
			d_mean = tf.broadcast_to(self.d_mean_marginal, tf.shape(inputs['mu']))
			position_sample = self.prior_position.sample(d_mean, nsample = nsample) # (n, B, L, 1) or (n, B, L, A)
		d_trans = tf.broadcast_to(self.d_mean_marginal, tf.shape(inputs['mu']))
		d_sd = tf.broadcast_to(self.d_sd_marginal, tf.shape(inputs['mu']))
		d = d_trans + d_sd * position_sample # (n, B, L, A)
		# rescaling d to (0, 1) as variant degree of selection, monotonic to d
		smax_sample, psmax = self.gene.sample(belong_mt) # (B, 1)
		smax_sample = tf.expand_dims(tf.expand_dims(smax_sample, -1), 0) # ( 1, B, 1, 1)
		s_sample = self.selection((smax_sample, tf.math.sigmoid(d))) # ( nsample, B, L, A)
		# marginal regularization
		logp_degree = -tfd.Normal(self.d_mean_marginal, self.d_sd_marginal).cross_entropy(tfd.Normal(d_trans, d_sd)) # (B, L, A)
		logp_degree = tf.reduce_sum(logp_degree * inputs['overlap_weight'] * (1 - inputs['refseq']))
		d_marginal = tfd.Normal(self.d_mean_marginal, self.d_sd_marginal).sample((rsample, 1, 1, 1)) # (rsample, 1, 1, 1)
		logp_marginal = tfd.Normal(d_trans, d_sd).log_prob(d_marginal) # (rsample, B, L, A)
#		logp_marginal = tfp.math.clip_by_value_preserve_gradient(logp_marginal, -10., 0.,)
		w = tf.clip_by_value(inputs['overlap_weight'] * (1 - inputs['refseq']), 1e-9, 1.)
		logp_marginal = tf.math.reduce_logsumexp(logp_marginal + tf.math.log(w), axis = [-1, -2]) - tf.math.log(tf.reduce_sum(w, axis = [-1, -2])) # (rsample, B)
		logp_marginal = tf.reduce_sum(tf.math.reduce_mean(logp_marginal, axis = 0))
		# ac loss
		if setting['use_syn'] == False:
			logp_ac_sample = self.popmodel((tf.math.log(inputs['mu']), s_sample, inputs['AN'], inputs['AC'])) * (1 - inputs['refseq']) * inputs['AA_mask']
		else:
			logp_ac_sample = self.popmodel((tf.math.log(inputs['mu']), s_sample, inputs['AN'], inputs['AC'])) * inputs['AA_mask']
		logp_ac = tfp.math.reduce_logmeanexp(logp_ac_sample, 0) # ( B, L, A)
		logp_ac = tf.reduce_sum(logp_ac, axis = [-1, -2]) # ( B)
		logp_ac = tf.reduce_sum(logp_ac)
		logp_ac_iw = tf.reduce_sum(psmax * tf.reduce_sum(inputs['overlap_weight'], axis = -2) / (belong_mt @ self.total_length)) + logp_ac
		return logp_ac, logp_ac_iw, logp_degree, logp_marginal
	def train_step(self, data):
		inputs = data
		with tf.GradientTape() as tape:
			logp_ac, logp_ac_iw, logp_degree, logp_marginal = self.get_loss(inputs, training = True)
			ac_loss = -logp_ac / setting['batch_size']
			reg_loss = -logp_degree / setting['batch_size']
			marginal_loss = -logp_marginal / setting['batch_size']
			total_loss = -logp_ac_iw * setting['training_weight']['ac'] / setting['batch_size'] + \
			             reg_loss * setting['training_weight']['reg'] + \
			             marginal_loss * setting['training_weight']['marginal']
		grads = tape.gradient(total_loss, self.trainable_weights)
		self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
		self.ac_loss_tracker.update_state(ac_loss)
		self.reg_loss_tracker.update_state(reg_loss)
		self.marginal_loss_tracker.update_state(marginal_loss)
		self.total_loss_tracker.update_state(total_loss)
		return {
			"ac_loss": self.ac_loss_tracker.result(),
			"reg_loss": self.reg_loss_tracker.result(),
			"marginal_loss": self.marginal_loss_tracker.result(),
			"total_loss": self.total_loss_tracker.result(),
		}
	@property
	def metrics(self):
		return [self.ac_loss_tracker, self.reg_loss_tracker, self.marginal_loss_tracker, self.total_loss_tracker]

class LRSchedule(tfk.optimizers.schedules.LearningRateSchedule):
	def __init__(self, initial_learning_rate, rate_per_entry = 0.999985, final_learning_rate = 1e-5, initial_entries = 5000):
		self.initial_learning_rate = initial_learning_rate
		self.final_learning_rate = final_learning_rate
		self.initial_steps = initial_entries // setting['batch_size']
		self.rate_per_step = rate_per_entry ** setting['batch_size']
	def __call__(self, step):
		changable_step = tf.where(tf.math.greater(step, self.initial_steps), step - self.initial_steps, 0)
		lr = self.initial_learning_rate * (self.rate_per_step ** changable_step)
		lr = tf.where(tf.math.greater(lr, self.final_learning_rate), lr, self.final_learning_rate)
		return lr

def create(saved_weights_path = None):
	data = gen_data(2, batch_size = 1)
	with mirrored_strategy.scope():
		model = ProbModel()
		_ = model(next(iter(data)))
		if saved_weights_path is not None:
			model.load_weights(tf.train.latest_checkpoint(saved_weights_path))
		else:
			if exists(f"{log_dir}/ckpt/checkpoint"):
				model.load_weights(tf.train.latest_checkpoint(f"{log_dir}/ckpt/"))
			else:
				print(model.summary())
	return model

def train(model, train_list, lr = 1e-3, patience = 3, epochs = 20, checkpoint = 0, clipnorm = None, clipvalue = None):
	data_size = get_data_size(train_list)
	data = gen_data(train_list, shuffle = True, batch_size = setting['batch_size'] * len(setting['GPU']))
	trainable = setting['trainable']
	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
	lr_schedule = LRSchedule(lr)
	stop_callback = tf.keras.callbacks.EarlyStopping(monitor='total_loss', patience = patience)
	model.gene.trainable = trainable['gene']
	print(f"training set: {train_list}; learning_rate: {lr}; ")
	print(f"trainable layers: {trainable}")
	print(f"training weight: {setting['training_weight']}")
	dist_data = mirrored_strategy.experimental_distribute_dataset(data)
	with mirrored_strategy.scope():
#		model.compile(optimizer=tfa.optimizers.AdamW(
		model.compile(optimizer=tf.optimizers.Adam(
#			weight_decay = setting['weight_decay'],
			learning_rate = lr_schedule, 
#			exclude_from_weight_decay = ["normalization", "gene"],
			clipnorm = clipnorm,
			clipvalue = clipvalue
		))
	model.fit(dist_data, callbacks = [stop_callback], epochs = epochs, steps_per_epoch = int(data_size / setting['batch_size'] / len(setting['GPU'])))
#	model.fit(data, callbacks = [stop_callback], epochs = epochs)
	model.save_weights(f"{log_dir}/ckpt/checkpoint_{checkpoint}")
	print(f"saved checkpoint: {checkpoint}")

def get_gene():
	model = create(f"{log_dir}/ckpt/")
	np.save(f"{log_dir}/smax_mean_gene.npy", model.gene.smax_mean_gene.numpy())

def output(test_list):
	model = create(f"{log_dir}/ckpt/")
	output_d_dir = f"{log_dir}/d_temp/"
	output_s_dir = f"{log_dir}/s_temp/"
	if not os.path.exists(output_d_dir):
		os.makedirs(output_d_dir)
	if not os.path.exists(output_s_dir):
		os.makedirs(output_s_dir)
	data = gen_data(test_list, shuffle = False, batch_size = setting['batch_size'] * len(setting['GPU']), repeat = 1)
	for inputs in iter(data):
		d, s, _, _, _ = model.call(inputs)
		for index in range(tf.shape(inputs['id'])[0].numpy()):
			uniprot_id = inputs['uniprot'][index].numpy().decode('utf-8')
			frac = inputs['frac'][index].numpy()
			damage = tf.math.sigmoid(d[index,:,:]) # (L, A)
			selection = s[index,:,:] # (L, A)
			weights = inputs['overlap_weight'][index]
			weighted_d = damage * weights
			weighted_s = selection * weights
			l = inputs['l'][index].numpy()
			np.save(f"{output_d_dir}/{uniprot_id}_{frac}.npy", weighted_d[:l].numpy())
			np.save(f"{output_s_dir}/{uniprot_id}_{frac}.npy", weighted_s[:l].numpy())

def combine(test_list_num, varname = "d", varname_df = "model_damage"):
	dataset = pd.read_table(os.path.expanduser(f"{setting['data_folder']}/list_{test_list_num}.txt"))
	output_dir = f"{log_dir}/combined_scores/"
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	for uniprot_id in dataset['UniprotID'].unique():
		dataset_sub = dataset[dataset['UniprotID']==uniprot_id]
		full_score = np.zeros((dataset_sub['end'].max(), len(AA_table)))
		for _, row in dataset_sub.iterrows():
			frac = row['frac']
			start = row['start']
			end = row['end']
			weighted_score = np.load(f"{log_dir}/{varname}_temp/{uniprot_id}_{frac}.npy")
			full_score[(start-1):end] += weighted_score
		df = pd.DataFrame(full_score, columns = list(setting['AA_list']))
		df['Protein_position'] = [i+1 for i in range(full_score.shape[0])]
		df = pd.melt(df, id_vars = "Protein_position", var_name = "AA_alt", value_name = varname_df)
		df = df.sort_values(by = ['Protein_position', 'AA_alt']).reset_index(drop = True)
		filename = f"{output_dir}/{uniprot_id}.txt.gz"
		if os.path.exists(filename):
			old_df = pd.read_table(filename)
			old_df[varname_df] = df[varname_df]
			old_df.to_csv(filename, index = False, sep = "\t")
		else:
			df.to_csv(filename, index = False, sep = "\t")	

