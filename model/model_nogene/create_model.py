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

class Selection(tfkl.Layer):
	def __init__(self, smin = -13.8, smax = 0., **kwargs):
		super().__init__(**kwargs)
		self.smin = smin
		self.smax = smax
	def call(self, degree):
		s = degree * (self.smax - self.smin) + self.smin # ( nsample, B, L, A) or (B, L, A)
		return(s)


class ProbModel(tfk.Model):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		geneset = pd.read_table(setting["geneset"], sep = "\t")
		self.total_length = tf.constant(geneset[['Length']], dtype = tf.float32) # (ngene, 1)
		self.prior_position = PriorPosition(use_cov = setting['use_cov'], aa_base = setting['aa_base_sampling'])
		self.selection = Selection(smin = setting['min_log_s'], smax = 0.)
		self.popmodel = PopACModel()
		self.d_mean_marginal = tf.Variable(setting['d_mean_marginal'], name = "d_mean", dtype = tf.float32)
		self.d_sd_marginal = tf.Variable(setting['d_sd_marginal'], name = "d_sd", dtype = tf.float32, constraint = lambda z: tfp.math.clip_by_value_preserve_gradient(z, 1e-3, setting['d_sd_max']))
		self.ac_loss_tracker = tfk.metrics.Mean(name="ac_loss")
	def call(self, inputs, training = None):
		d_trans = tf.broadcast_to(self.d_mean_marginal, tf.shape(inputs['mu']))
		d_sd = tf.broadcast_to(self.d_sd_marginal, tf.shape(inputs['mu']))
		s = self.selection(tf.math.sigmoid(d_trans)) # (B, L, A)
		return d_trans, s, d_sd  # (B, L, A)
	def get_loss(self, inputs, training = None, msample = setting['msample'], nsample = setting['nsample'], rsample = setting['rsample']):
		if self.prior_position.use_cov:
			position_sample = self.prior_position.sample(inputs['contact_decomp'], nsample = nsample) # (n, B, L, 1)
		else:
			d_mean = tf.broadcast_to(self.d_mean_marginal, tf.shape(inputs['mu']))
			position_sample = self.prior_position.sample(d_mean, nsample = nsample) # (n, B, L, 1) or (n, B, L, A)
		d_trans = tf.broadcast_to(self.d_mean_marginal, tf.shape(inputs['mu']))
		d_sd = tf.broadcast_to(self.d_sd_marginal, tf.shape(inputs['mu']))
		d = d_trans + d_sd * position_sample # (n, B, L, A)
		# rescaling d to (0, 1) as variant degree of selection, monotonic to d
		s_sample = self.selection(tf.math.sigmoid(d)) # ( nsample, B, L, A)
		# ac loss
		if setting['use_syn'] == False:
			logp_ac_sample = self.popmodel((tf.math.log(inputs['mu']), s_sample, inputs['AN'], inputs['AC'])) * (1 - inputs['refseq']) * inputs['AA_mask']
		else:
			logp_ac_sample = self.popmodel((tf.math.log(inputs['mu']), s_sample, inputs['AN'], inputs['AC'])) * inputs['AA_mask']
		logp_ac = tfp.math.reduce_logmeanexp(logp_ac_sample, 0) # ( B, L, A)
		logp_ac = tf.reduce_sum(logp_ac, axis = [-1, -2]) # ( B)
		logp_ac = tf.reduce_sum(logp_ac)
		return logp_ac
	def train_step(self, data):
		inputs = data
		with tf.GradientTape() as tape:
			logp_ac = self.get_loss(inputs, training = True)
			ac_loss = -logp_ac / setting['batch_size']
		grads = tape.gradient(ac_loss, self.trainable_weights)
		self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
		self.ac_loss_tracker.update_state(ac_loss)
		return {
			"ac_loss": self.ac_loss_tracker.result(),
			"d_mean": self.d_mean_marginal,
			"d_sd": self.d_sd_marginal
		}
	@property
	def metrics(self):
		return [self.ac_loss_tracker]

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
	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
	lr_schedule = LRSchedule(lr)
	stop_callback = tf.keras.callbacks.EarlyStopping(monitor='total_loss', patience = patience)
	print(f"training set: {train_list}; learning_rate: {lr}; ")
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

