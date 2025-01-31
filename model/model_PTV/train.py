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
from build_feature import gen_data

with open("setting.json", "r") as setting_file:
	setting = json.load(setting_file)

gpus = tf.config.list_physical_devices('GPU')
used_gpus = [gpus[i] for i in setting['GPU']]
tf.config.set_visible_devices(used_gpus, 'GPU')

class PriorGene(tfkl.Layer):
	def __init__(self, smin, init_smax_mean, init_smax_sd, **kwargs):
		super().__init__(**kwargs)
		self.smin = smin
		# prior
		self.smax_mean_global = tf.Variable(init_smax_mean, name = "smax_mean_global", dtype = tf.float32, constraint = lambda z: tfp.math.clip_by_value_preserve_gradient(z, 1e-3 + smin, 0.))
		self.smax_sd_global = tf.Variable(init_smax_sd, name = "smax_sd_global", dtype = tf.float32, constraint = lambda z: tfp.math.clip_by_value_preserve_gradient(z, 1e-5, 3.))
	def call(self):
		return self.smax_mean_global, self.smax_sd_global

class Gene(tfkl.Layer):
	def __init__(self, ngene, smin, init_smax_mean, init_smax_sd, **kwargs):
		super().__init__(**kwargs)
		self.ngene = ngene
		self.smin = smin
		# posterior
		self.smax_mean_gene = tf.Variable(tf.ones((ngene, 1)) * init_smax_mean, name = "smax_mean_gene", dtype = tf.float32, constraint = lambda z: tfp.math.clip_by_value_preserve_gradient(z, 1e-3 + smin, 0.))
	def call(self, belong_mt):
		smax_mean = belong_mt @ self.smax_mean_gene
		return smax_mean
	def sample(self, belong_mt, smax_mean_global, smax_sd_global, msample, from_prior = False):
		# belong_mt: (b, ngene)
		smax_global_distr = tfd.Normal(smax_mean_global, smax_sd_global)
		if from_prior == False:
			smax_gene_sample = tf.expand_dims(self.smax_mean_gene, 0) # (1, ngene, 1)
		else:
			smax_gene_sample = smax_global_distr.sample((msample, self.ngene, 1))
		smax_sample = belong_mt @ smax_gene_sample # (msample, b, 1)
		psmax = belong_mt @ smax_global_distr.log_prob(smax_gene_sample)
		return smax_sample, psmax

class Selection(tfkl.Layer):
	def __init__(self, smin = -13.8, **kwargs):
		super().__init__(**kwargs)
		self.smin = smin
	def call(self, inputs):
		smax_sample = tfp.math.clip_by_value_preserve_gradient(tf.expand_dims(inputs, -1), 1e-3 + self.smin, 0.)
		return smax_sample # (msample, b, 1, 1)

class ProbModel(tfk.Model):
	def __init__(self, smin = setting['min_log_s'], ngene = setting['ngene'], init_smax_mean = -5.06, init_smax_sd = 2.82, **kwargs):
		super().__init__(**kwargs)
		self.ngene = ngene
		self.priorgene = PriorGene(smin = smin, init_smax_mean = init_smax_mean, init_smax_sd = init_smax_sd)
		self.gene = Gene(ngene = ngene, smin = smin, init_smax_mean = init_smax_mean, init_smax_sd = init_smax_sd)
		self.selection = Selection(smin = smin)
		self.popmodel = PopACModel()
		self.ac_loss_tracker = tfk.metrics.Mean(name="ac_loss")
		self.total_loss_tracker = tfk.metrics.Mean(name="total_loss")
		self.train_prior_only = True
	def sample(self, inputs, msample = 1, from_prior = False):
		belong_mt = tf.one_hot(inputs['id'], depth = self.ngene) # (b, ngene)
		smax_mean_global, smax_sd_global = self.priorgene.call()
		smax_sample, psmax = self.gene.sample(belong_mt, smax_mean_global, smax_sd_global, msample, from_prior)
		s_sample = self.selection(smax_sample)
		# inputs: (b, v, 1)
		logps = self.popmodel((tf.math.log(inputs['mu']), s_sample, inputs['AN'], inputs['AC'])) # (msample, b, v, 1)
		logps = inputs['mask'] * logps
		logps = tf.math.reduce_sum(logps, axis = -2) # (msample, b, 1)
		logps = tfp.math.reduce_logmeanexp(logps, axis = 0) # (b, 1)
		psmax = tf.math.reduce_sum(psmax, axis = 0)
		return logps, psmax # (b, 1)
	def train_step(self, data):
		inputs = data
		with tf.GradientTape() as tape:
			if self.train_prior_only:
				logps, psmax = self.sample(inputs, msample = setting['msample'], from_prior = self.train_prior_only)
				psmax = 0.
			else:
				logps, psmax = self.sample(inputs, msample = 1, from_prior = self.train_prior_only)
			ac_loss = -tf.reduce_sum(logps) / setting['batch_size']
			total_loss = -tf.reduce_sum(logps + psmax * inputs['weight'][:, tf.newaxis]) / setting['batch_size']
		grads = tape.gradient(total_loss, self.trainable_weights)
		self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
		self.ac_loss_tracker.update_state(ac_loss)
		self.total_loss_tracker.update_state(total_loss)
		result = {
			"ac_loss": self.ac_loss_tracker.result(),
			"total_loss": self.total_loss_tracker.result(),
		}
		if self.train_prior_only:
			result['smax_mean_global'] = self.priorgene.smax_mean_global
			result['smax_sd_global'] = self.priorgene.smax_sd_global
		else:
			result['max_smax_mean_gene'] = tf.reduce_max(self.gene.smax_mean_gene)
			result['min_smax_mean_gene'] = tf.reduce_min(self.gene.smax_mean_gene)
		return result
	@property
	def metrics(self):
		return [self.ac_loss_tracker, self.total_loss_tracker]

class LRSchedule(tfk.optimizers.schedules.LearningRateSchedule):
	def __init__(self, initial_learning_rate, rate_per_entry = 0.99999, final_learning_rate = 1e-5, initial_entries = 10000):
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
	model = ProbModel()
	data = gen_data()
	_ = model.sample(next(iter(data)))
	if saved_weights_path is not None:
		model.load_weights(saved_weights_path)
	return model, data

def train(model, data, epochs = 50, init_lr = 1e-2, lr_decay_epoch = 20, lr_warmup_epoch = 5, lr_decay_rate = 0.1, prior = True, post = False):
	def epochscheduler(epoch, lr):
		if epoch < lr_warmup_epoch:
			return lr
		else:
			return lr * tf.math.pow(lr_decay_rate, 1/lr_decay_epoch)
	model.train_prior_only = not post
	model.priorgene.trainable = prior
	model.gene.trainable = post
	lr_callback = tf.keras.callbacks.LearningRateScheduler(epochscheduler)
	model.compile(optimizer=tfk.optimizers.Adam(learning_rate = init_lr))
	model.fit(data, callbacks=[lr_callback], epochs = epochs)

def main():
	model, data = create()
	#train(model, data, epochs = 15, init_lr = 0.03, lr_warmup_epoch = 0, lr_decay_epoch = 15, prior = True, post = False)
	log_dir = "result"
	#log = open(f"{log_dir}/par.log", "w")
	#print(f"smax_mean_global: {model.priorgene.smax_mean_global.numpy()}", file = log)
	#print(f"smax_sd_global: {model.priorgene.smax_sd_global.numpy()}", file = log)
	#log.close()
	model.gene.smax_mean_gene.assign(tf.ones((model.ngene, 1)) * model.priorgene.smax_mean_global)
	model.save_weights(f"{log_dir}/ckpt/checkpoint_0")
	train(model, data, epochs = 50, init_lr = 0.1, lr_warmup_epoch = 0, lr_decay_epoch = 25, prior = False, post = True)
	model.save_weights(f"{log_dir}/ckpt/checkpoint_1")
	np.save(f"{log_dir}/smax_mean_gene.npy", model.gene.smax_mean_gene.numpy())

if __name__=="__main__":
	main()

