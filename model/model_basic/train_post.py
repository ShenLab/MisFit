from create_model import *

class PostSelection(tfk.Model):
	def __init__(self, smin, mean_layer = [10, 10], sd_layer = [10, 10], **kwargs):
		super().__init__(**kwargs)
		self.smin = smin
		self.post_mean = tfk.Sequential()
		for dim in mean_layer:
			self.post_mean.add(Dense(dim, activation = "relu"))
		self.post_mean.add(Dense(1))
		self.post_sd = tfk.Sequential()
		for dim in sd_layer:
			self.post_sd.add(tfkl.Dense(dim, activation = "relu"))
		self.post_sd.add(tfkl.Dense(1, activation = "softplus"))
	def call(self, inputs, d_mean, d_sd, s_mean, s_sd):
		an = tf.math.log(inputs['AN'] + 0.1) # B, L, A
		ac = tf.math.log(inputs['AC'] + 0.1)
		mu = tf.math.log(inputs['mu'] + 1e-10)
		# s_mean, s_sd : B, 1, 1, d_mean, d_sd: B, L, A
		x = tf.stack([mu, an, ac - an, tf.broadcast_to(s_mean, tf.shape(mu)), tf.broadcast_to(s_sd, tf.shape(mu)), tf.broadcast_to(d_mean, tf.shape(mu)), tf.broadcast_to(d_sd, tf.shape(mu))], axis = -1) # (B, L, A, ?)
		d_post_mean = self.post_mean(x)
		d_post_mean = tf.squeeze(d_post_mean, -1)
		d_post_sd = self.post_sd(x)
		d_post_sd = tf.squeeze(d_post_sd, -1)
		return d_post_mean, d_post_sd # (B, L, A)
	def get_s(self, inputs, d_mean, d_sd, s_mean, s_sd):
		d_post_mean, _ = self.call(inputs, d_mean, d_sd, s_mean, s_sd)
		s_post_mean = tf.math.sigmoid(d_post_mean) * (s_mean - self.smin) + self.smin
		return s_post_mean
	def sample(self, inputs, d_mean, d_sd, s_mean, s_sd):
		d_post_mean, d_post_sd = self.call(inputs, d_mean, d_sd, s_mean, s_sd)
		d_post_distr = tfd.Normal(d_post_mean, d_post_sd)
		d_post_sample = d_post_distr.sample() # (B, L, A)
		kl_d = d_post_distr.kl_divergence(tfd.Normal(d_mean, d_sd))
		smax_post_sample = tfd.Normal(s_mean, s_sd).sample() # B, 1, 1
		s_post_sample = tf.math.sigmoid(d_post_sample) * (smax_post_sample - self.smin) + self.smin # B, L, A
		return s_post_sample, kl_d
		
class PostModel(tfk.Model):
	def __init__(self, probmodel, smin = setting['min_log_s'], mean_layer = setting['post_mean_layer'], sd_layer = setting['post_sd_layer'], **kwargs):
		super().__init__(**kwargs)
		self.probmodel = probmodel
		self.probmodel.trainable = False
		self.post_selection = PostSelection(smin = smin, mean_layer = mean_layer, sd_layer = sd_layer)
		self.popmodel = PopACModel()
		self.ac_loss_tracker = tfk.metrics.Mean(name="ac_loss")
		self.total_loss_tracker = tfk.metrics.Mean(name="total_loss")
	def call(self, inputs, nogene = False):
		d_mean, _, d_sd, s_mean, s_sd = self.probmodel.call(inputs)
		if nogene:
			s_mean = tf.broadcast_to(setting['smax_mean_global'], tf.shape(s_mean))
			s_sd = tf.broadcast_to(setting['smax_sd_global'], tf.shape(s_sd))
		s_post_mean = self.post_selection.get_s(inputs, d_mean, d_sd, s_mean, s_sd)
		return tf.math.sigmoid(d_mean), s_post_mean
	def get_loss(self, inputs, nogene = False):
		d_mean, _, d_sd, s_mean, s_sd = self.probmodel.call(inputs)
		if nogene:
			s_mean = tf.broadcast_to(setting['smax_mean_global'], tf.shape(s_mean))
			s_sd = tf.broadcast_to(setting['smax_sd_global'], tf.shape(s_sd))
		s_post_sample, kl_d = self.post_selection.sample(inputs, d_mean, d_sd, s_mean, s_sd)
		logps = self.popmodel((tf.math.log(inputs['mu']), s_post_sample, inputs['AN'], inputs['AC'])) * inputs['AA_mask']
		return logps, kl_d
	def train_step(self, inputs):
		with tf.GradientTape() as tape:
			logps, kl = self.get_loss(inputs, nogene = setting['nogene'])
			ac_loss = -tf.reduce_sum(logps * inputs['overlap_weight']) / setting['batch_size']
			total_loss = ac_loss + tf.reduce_sum(kl * inputs['overlap_weight']) / setting['batch_size']
		grads = tape.gradient(total_loss, self.trainable_weights)
		self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
		self.ac_loss_tracker.update_state(ac_loss)
		self.total_loss_tracker.update_state(total_loss)
		result = {
			"ac_loss": self.ac_loss_tracker.result(),
			"total_loss": self.total_loss_tracker.result(),
		}
		return result
	@property
	def metrics(self):
		return [self.ac_loss_tracker, self.total_loss_tracker]

def create_post(saved_weights_path = None):
	data = gen_data(2, batch_size = 2)
	with mirrored_strategy.scope():
		model = ProbModel()
		inputs = next(iter(data))
		_ = model(inputs)
		if saved_weights_path is not None:
			model.load_weights(tf.train.latest_checkpoint(saved_weights_path))
		else:
			if exists(f"{log_dir}/ckpt/checkpoint"):
				model.load_weights(tf.train.latest_checkpoint(f"{log_dir}/ckpt/"))
		model.trainable = False
		post_model = PostModel(probmodel = model)
		_ = post_model.call(inputs)
		if exists(f"{log_dir}/ckpt_post/checkpoint"):
			post_model.post_selection.load_weights(tf.train.latest_checkpoint(f"{log_dir}/ckpt_post/"))
	return post_model

def train_post(model, train_list, lr = 1e-3, patience = 3, epochs = 20, checkpoint = 0, clipnorm = None, clipvalue = None):
	data_size = get_data_size(train_list)
	data = gen_data(train_list, shuffle = True, batch_size = setting['batch_size'] * len(setting['GPU']))
	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
	lr_schedule = LRSchedule(lr)
	stop_callback = tf.keras.callbacks.EarlyStopping(monitor='total_loss', patience = patience, restore_best_weights = True)
	dist_data = mirrored_strategy.experimental_distribute_dataset(data)
	with mirrored_strategy.scope():
		model.compile(optimizer=tf.optimizers.Adam(
			learning_rate = lr_schedule, 
			clipnorm = clipnorm,
			clipvalue = clipvalue
		))
	print(f"training posterior model")
	model.fit(dist_data, callbacks = [stop_callback], epochs = epochs, steps_per_epoch = int(data_size / setting['batch_size'] / len(setting['GPU'])))
	model.post_selection.save_weights(f"{log_dir}/ckpt_post/checkpoint_{checkpoint}")
	print(f"saved checkpoint: {checkpoint}")

def output_post(test_list, output_d = False, nogene = setting['nogene']):
	model = create_post()
	output_d_dir = f"{log_dir}/d_temp/"
	output_s_dir = f"{log_dir}/s_temp/"
	if not os.path.exists(output_d_dir):
		os.makedirs(output_d_dir)
	if not os.path.exists(output_s_dir):
		os.makedirs(output_s_dir)
	data = gen_data(test_list, shuffle = False, batch_size = setting['batch_size'] * len(setting['GPU']), repeat = 1)
	for inputs in iter(data):
		d, s = model.call(inputs, nogene = nogene)
		for index in range(tf.shape(inputs['id'])[0].numpy()):
			uniprot_id = inputs['uniprot'][index].numpy().decode('utf-8')
			frac = inputs['frac'][index].numpy()
			weights = inputs['overlap_weight'][index]
			l = inputs['l'][index].numpy()
			selection = s[index,:,:] # (L, A)
			weighted_s = selection * weights
			np.save(f"{output_s_dir}/{uniprot_id}_{frac}.npy", weighted_s[:l].numpy())
			if output_d:
				damage = d[index,:,:] # (L, A)
				weighted_d = damage * weights
				np.save(f"{output_d_dir}/{uniprot_id}_{frac}.npy", weighted_d[:l].numpy())

def main():
	model = create_post()
	train_post(model, [2], epochs = 30, patience = 5)

if __name__=="__main__":
	main()

