import numpy as np
import pandas as pd
import argparse
import logging

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("-mu", "--mu", action="store", type=float, help="mutation rate")
	args = parser.parse_args()
	return args

def main():
	# parse arguments
	args = parse_args()

	# effective population size of time
	T0 = 144740
	T1 = 3880
	T2 = 1120
	T3 = 716
	gr3 = 0.00307
	T4 = 205
	gr4 = 0.03
	N = np.zeros(T0+T1+T2+T3+T4, dtype=int)
	N[0:(T0+T1)] = 14474
	N[(T0+T1):(T0+T1+T2)] = 1861
	N[T0+T1+T2] = 1032
	for t in range(1, T3):
		N[T0+T1+T2+t] = N[T0+T1+T2+t-1] * (1 + gr3)
	for t in range(T4):
		N[T0+T1+T2+T3+t] = N[T0+T1+T2+T3+t-1] * (1 + gr4)

	# parameters
	mu = args.mu
	L = int(1e5)
	filename = "sim_q/simulation_back_" + str(mu) + ".csv"

	# simulation
	for s in [5e-6, 1.6e-5, 5e-5, 1.6e-4, 5e-4, 1.6e-3, 5e-3, 1.6e-2, 5e-2, 1.6e-1, 5e-1, 1, 0.6, 0.7, 0.8, 0.9]:
		print("mu =", mu, " s =", s)
		AC = np.zeros(L)
		AF = AC / (2 * N[0])
		for t in range(1, len(N) - 1):
			exist_AC = np.random.binomial(2 * N[t], AF * (1 - s))
			back_AC = np.random.binomial(exist_AC, mu)
			new_AC = np.random.binomial(2 * N[t] - exist_AC, mu)
			AC = exist_AC + new_AC - back_AC
			AF = AC / (2 * N[t])
			if (t%5000==0):
				print(t, "generations processed.", flush = True)
		AFadj = AF * (1 - s) * (1 - 2 * mu) + mu
		#(AC_uni, Occ) = np.unique(np.array(AC), return_counts = True)
		(AF_uni, Occ) = np.unique(np.array(AFadj), return_counts = True)
		df = pd.DataFrame({
			"mu": [mu] * len(Occ),
			"s": [s] * len(Occ),
			"AF_uni": AF_uni, 
			"Occ": Occ

		})
		df.to_csv(filename, header=False, index=False, mode="a")

if __name__ == "__main__":
		main()

