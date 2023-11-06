import pandas as pd
import numpy as np

def main():
	summary_df = pd.DataFrame()
	mids = [1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7]
	logmids = [np.log(mid) for mid in mids]
	logbins = [(logmids[i] + logmids[i+1]) / 2 for i in range(len(mids)-1)]
	bins = [np.exp(logbin) for logbin in logbins]
	bins = [0.] + bins + [1.]
	mu_name = "roulette_mu"
	df = pd.read_table("syn_sample.txt.gz")
	for i in range(len(bins) - 1):
		upper = bins[i+1]
		lower = bins[i]
		mu = mids[i]
		prob = len(df[(df[mu_name] > lower) & (df[mu_name] <= upper)]) / len(df)
		summary_df = pd.concat([summary_df, pd.DataFrame({'mu': [mu], 'lower': [lower], 'upper' : [upper], 'weight': [prob]})])
	
	summary_df.to_csv("mu_summary.txt", sep = "\t", index = False)

if __name__=="__main__":
	main()
