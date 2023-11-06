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

def main():
	# parse arguments
	args = parse_args()
	mu0 = 1e-8
	final_ne = 1.5e6 # final Ne
	clip_gen = int(1e4) # history length
	N = get_history(final_ne = final_ne, clip = clip_gen)

	# parameters
	mu = args.mu
	L = int(1e5)
	filename = "sim_q/simulation_" + str(mu) + ".csv"

	# simulation
	for s in [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 0.5, 0.7, 0.9]:
		s_homo = np.minimum(2*s, 1)
		print("mu =", mu, " s =", s)
		AF_sample = np.zeros(L)
		for t in range(len(N)):
			AF_inh = ((1-s)*(1-AF_sample)*AF_sample+(1-s_homo)*AF_sample*AF_sample)/((1-AF_sample)**2+2*(1-s)*(1-AF_sample)*AF_sample+(1-s_homo)*AF_sample*AF_sample)
			AF = AF_inh * (1 - mu0) + (1 - AF_inh) * mu
			AC = np.random.binomial(2 * N[t], AF)
			AF_sample = AC / (2*N[t])
		AF = ((1-s)*(1-AF)*AF+(1-s_homo)*AF*AF)/((1-AF)**2+2*(1-s)*(1-AF)*AF+(1-s_homo)*AF*AF)
		(AF_uni, Occ) = np.unique(np.array(AF), return_counts = True)
		df = pd.DataFrame({
			"mu": [mu] * len(Occ),
			"s": [s] * len(Occ),
			"AF_uni": AF_uni, 
			"Occ": Occ
		})
		df.to_csv(filename, header=False, index=False, mode="a")

if __name__ == "__main__":
		main()

