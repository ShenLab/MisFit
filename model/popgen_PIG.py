import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tf.keras
tfkl = tf.keras.layers

# Poisson Inverse Gaussian log probability
def pinvgauss_logp(pig_mu, pig_lambda, k):
	term0 = 0.5 * tf.math.log(2 * pig_lambda / np.pi) + pig_lambda / pig_mu - tf.math.lgamma(k + 1)
	term1 = - 0.5 * (k - 0.5) * tf.math.log(2 / pig_lambda + 1 / pig_mu ** 2)
	bessel_x = tf.math.sqrt(2 * pig_lambda + pig_lambda ** 2 / pig_mu ** 2)
	term2 = tfp.math.log_bessel_kve(k - 0.5, bessel_x) - bessel_x
	return term0 + term1 + term2

def f_log_ig_mu(logmu, logs): # log mu, log s --> log IG mu (-inf, 0)
	logneuconst = 11.49612
	log_ig_mu = -tf.math.softplus(logs + logneuconst) + logneuconst + logmu
	return log_ig_mu

def f_log_ig_lambda(logmu, logs): # log mu, log s --> log IG lambda (-inf, inf)
	c = 4.078826
	k = 6.762855
	a = 1.781733
	b = 15.63719
	log_ig_lambda = c * tf.math.exp(k * logs) + a * logmu + b
	return log_ig_lambda

class PopACModel(tfkl.Layer):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		
	def call(self, inputs): # inputs: (log mu, log s, an, ac) --> logp
		logmu, logs, an, ac = inputs
		logmu = tf.clip_by_value(logmu, -22., -9.)
		logs = tfp.math.clip_by_value_preserve_gradient(logs, -13., 0.)
		log_ig_mu = f_log_ig_mu(logmu, logs)
		log_ig_lambda = f_log_ig_lambda(logmu, logs)
		an = tf.clip_by_value(an, 1., tf.float32.max)
		logp = pinvgauss_logp(tf.math.exp(log_ig_mu) * an, tf.math.exp(log_ig_lambda) * an, ac)
		return logp

