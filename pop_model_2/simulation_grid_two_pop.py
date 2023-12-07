import numpy as np
import pandas as pd
import argparse
import logging
import sys
from generate_history import get_history

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("-mu", "--mu", action="store", type=float, help="mutation rate")
	args = parser.parse_args()
	return args

def adjust_AF(AF, s, s_homo):
	AF_adj = ((1-s)*(1-AF)*AF+(1-s_homo)*AF*AF)/((1-AF)**2+2*(1-s)*(1-AF)*AF+(1-s_homo)*AF*AF)
	return AF_adj

def sim_one_gen(AF_sample, s, s_homo, Ne, mu, mu0):
	AF_inh = adjust_AF(AF_sample, s, s_homo)
#	AF_inh = ((1-s)*(1-AF_sample)*AF_sample+(1-s_homo)*AF_sample*AF_sample)/((1-AF_sample)**2+2*(1-s)*(1-AF_sample)*AF_sample+(1-s_homo)*AF_sample*AF_sample)
	AF = AF_inh * (1 - mu0) + (1 - AF_inh) * mu
	AC = np.random.binomial(2 * Ne, AF)
	AF_sample = AC / (2*Ne)
	return AF_sample, AF

def main():
	# parse arguments
	args = parse_args()
	mu0 = 1e-8
	final_ne = 1.5e6 # final Ne
	clip_gen = int(1e4) # history length
	split_gens = [2000, 200, 20, 0] # split length
	N = get_history(final_ne = final_ne, clip = clip_gen)

	# parameters
	mu = args.mu
	L = int(1e5)
	filename = "sim_q_two_pop/simulation_" + str(mu) + ".csv"

	# simulation
	for s in [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.9]:
		s_homo = np.minimum(2*s, 1)
		print("mu =", mu, " s =", s)
		AF_samples = [np.zeros(L)]
		t = 0
		split_index = 0
		while t < len(N) - 1:
			if t >= len(N) - split_gens[split_index]:
				AF_samples.append(AF_samples[-1].copy())
				split_index += 1
			for i, AF_sample in enumerate(AF_samples):
				AF_sample, _ = sim_one_gen(AF_sample, s, s_homo, N[t], mu, mu0)
				AF_samples[i] = AF_sample
			t += 1
		AF_dict = {
			"mu": [mu] * L,
			"s": [s] * L,
		}
		for i, AF_sample in enumerate(AF_samples):
			AF_sample, AF = sim_one_gen(AF_sample, s, s_homo, N[t], mu, mu0)
			AF_dict[f"AF_{i}"] = adjust_AF(AF, s, s_homo)
		df = pd.DataFrame(AF_dict)
		df.to_csv(filename, header=False, index=False, mode="a")

if __name__ == "__main__":
		main()

