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

def f_log_ig_mu(logmu, logit_s): # log mu, log s --> log IG mu (-inf, 0)
	logneuconst = 9.139235
	log_ig_mu = -tf.math.softplus(logit_s + logneuconst) + logneuconst + logmu
	return log_ig_mu

def f_log_ig_lambda(logmu): # log mu, log s --> log IG lambda (-inf, inf)
	c = 0.1725752
	d = 7.721602
	e = 65.32929
	log_ig_lambda = c * (logmu ** 2) + d * logmu + e
	return log_ig_lambda

class PopACModel(tfkl.Layer):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		
	def call(self, inputs): # inputs: (log mu, logit s, an, ac) --> logp
		logmu, logit_s, an, ac = inputs
		logmu = tf.clip_by_value(logmu, -22., -9.)
		logit_s = tfp.math.clip_by_value_preserve_gradient(logit_s, -13.82, 0.)
		log_ig_mu = f_log_ig_mu(logmu, logit_s)
		log_ig_lambda = f_log_ig_lambda(logmu)
		an = tf.clip_by_value(an, 1., tf.float32.max)
		logp = pinvgauss_logp(tf.math.exp(log_ig_mu) * an, tf.math.exp(log_ig_lambda) * an, ac)
		return logp

