import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.neighbors import KernelDensity

gpus = tf.config.list_physical_devices('GPU')
used_gpus = [gpus[i] for i in [0]]
tf.config.set_visible_devices(used_gpus, 'GPU')

def transform(x):
	return tf.clip_by_value(tf.math.sigmoid(x), 1e-6, 1.)

def main():
	result_dir = "result"
	nsample = 5000
	CI = 0.95
	log_s_mean = np.load(f"{result_dir}/smax_mean_gene.npy") # (ngene, 1)
	log_s_sd = np.load(f"{result_dir}/smax_sd_gene.npy") # (ngene, 1)
	geneset = pd.read_table("../../variant/list/geneset_uniprot_len.txt")
	geneset[['logit_s_mean']] = log_s_mean
	geneset[['logit_s_sd']] = log_s_sd
	s_distri = tfp.distributions.Normal(log_s_mean, log_s_sd)
	log_s_sample = s_distri.sample(nsample) # (nsample, ngene, 1)
	s_sample = transform(log_s_sample)
	s_mean = tf.reduce_mean(s_sample, axis = 0)
	s_lower = s_distri.quantile((1 - CI) / 2)
	s_lower = transform(s_lower)
	s_upper = s_distri.quantile((1 + CI) / 2)
	s_upper = transform(s_upper)
	s_median = transform(log_s_mean)

#	geneset['s_mode'] = pd.NA
	# kernel density for MAP
#	step = np.arange(-6, 0., 0.005, dtype = np.float32)
#	step = 10**step[:, np.newaxis]
#	for i in range(tf.shape(s_sample)[1]):
#		kde = KernelDensity(bandwidth = s_mean[i, 0].numpy() * 0.5).fit(s_sample[:, i, :])
#		density = kde.score_samples(step)
#		j = np.argmax(density)
#		geneset.at[i, 's_mode'] = step[j, 0]
#		if (i % 50 == 0):
#			print(i, j, step[j, 0])
	geneset[['s_mean']] = s_mean.numpy()
	geneset[['s_lower']] = s_lower.numpy()
	geneset[['s_upper']] = s_upper.numpy()
	geneset[['s_median']] = s_median.numpy()
	# gene with no coverage of AC
	exclude = set()
	with open("../../variant/snv_info_2/combine_covered_list_lof.txt", "r") as f:
		for line in f:
			exclude.add(line.strip())
#	geneset.loc[~geneset['UniprotID'].isin(exclude), ['logit_s_mean', 'logit_s_sd', 's_mean', 's_lower', 's_upper', 's_median', 's_mode']] = pd.NA
	geneset.loc[(~geneset['UniprotID'].isin(exclude))|(geneset['Chrom']=="Y"), ['logit_s_mean', 'logit_s_sd', 's_mean', 's_lower', 's_upper', 's_median']] = pd.NA
	geneset.to_csv(f"{result_dir}/geneset_lof_s.txt", index = False, sep = "\t")

if __name__=="__main__":
	main()
