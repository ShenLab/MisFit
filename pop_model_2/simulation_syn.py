import numpy as np
import pandas as pd
import argparse
import logging
import sys
from generate_history import get_history

def main(final_ne_M, mu_scale, clip_gen_10k, subset):
	if subset == "T":
		suffix = "_highmu"
	else:
		suffix = ""
	popnames = ['UKBB', 'gnomAD_NFE_exome', 'gnomAD_NFE_genome']
	s = 1e-6 # synonymous variant assumed selection coeff
	mu0 = 1e-8
	final_ne = float(final_ne_M) * 1e6 # final Ne
	clip_gen = int(float(clip_gen_10k) * 1e4) # history length
	N = get_history(final_ne = final_ne, clip = clip_gen)

	sample = pd.read_table(f"syn_sample{suffix}.txt.gz")
	sample['AN'] = 0
	sample['AC'] = 0
	for popname in popnames:
		sample['AN'] += sample[popname + "_AN"]
		sample['AC'] += sample[popname + "_AC"]
	sample = sample[(sample['AN'] > 0) & (sample['roulette_mu'] > 0)]
	mu = sample['roulette_mu'].to_numpy()
#	mu = sample['gnomAD_mu_old'].to_numpy()
	mu = mu * float(mu_scale)
	mu0 = mu0 * float(mu_scale)
	sample_AC = sample['AC']
	sample_AN = sample['AN']
	L = len(mu)

	filename = f"sim_q_syn_{clip_gen_10k}/simulation{suffix}_ne_{final_ne_M}_{mu_scale}.csv"

	# simulation
	AF_sample = 0
	s_homo = np.minimum(2*s, 1)
	
	for t in range(len(N)):
		AF_inh = ((1-s)*(1-AF_sample)*AF_sample+(1-s_homo)*AF_sample*AF_sample)/((1-AF_sample)**2+2*(1-s)*(1-AF_sample)*AF_sample+(1-s_homo)*AF_sample*AF_sample)
		AF = AF_inh * (1 - mu0) + (1 - AF_inh) * mu
		AC = np.random.binomial(2 * N[t], AF)
		AF_sample = AC / (2*N[t])
		if (t%10000==0):
			print(t, "generations processed.", flush = True)
	
	AF = ((1-s)*(1-AF)*AF+(1-s_homo)*AF*AF)/((1-AF)**2+2*(1-s)*(1-AF)*AF+(1-s_homo)*AF*AF)
	sample_AC_sim = np.random.binomial(sample_AN, AF)
	df = pd.DataFrame({
		"AN": sample_AN,
		"AC": sample_AC,
		"AF": sample_AC / sample_AN,
		"AC_sim": sample_AC_sim,
		"AF_sim": sample_AC_sim / sample_AN,
	})
	df.to_csv(filename, index=False)

if __name__ == "__main__":
	main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

