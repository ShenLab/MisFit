import pandas as pd
import numpy as np
import sys

def main(clip_gen_10k, subset):
	if subset == "T":
		suffix = "_highmu"
	else:
		suffix = ""
	summary_df = pd.DataFrame()
	bins = [-np.inf, 0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1]
	nes = [1, 1.5, 2, 3, 4, 5]
	scales = [1]
	for scale in scales:
		for ne in nes:
			df = pd.read_csv(f"sim_q_syn_{clip_gen_10k}/simulation{suffix}_ne_{ne}_{scale}.csv")
			df['AF_sim_adj_minor'] = 1 - df['AF_sim']
			df['AF_sim_adj_minor'] = df[['AF_sim', 'AF_sim_adj_minor']].min(axis = 1)
			df['AF_adj_minor'] = 1 - df['AF']
			df['AF_adj_minor'] = df[['AF', 'AF_adj_minor']].min(axis = 1)
		
			for i in range(len(bins) - 1):
				upper = bins[i+1]
				lower = bins[i]
				prob = len(df[(df['AF_sim'] > lower) & (df['AF_sim'] <= upper)]) / len(df)
				prob_minor = len(df[(df['AF_sim'] > lower) & (df['AF_sim'] <= min(upper, 0.5))]) / len(df[df['AF_sim'] <= 0.5])
				prob_adj_minor = len(df[(df['AF_sim_adj_minor'] > lower) & (df['AF_sim_adj_minor'] <= upper)]) / len(df)
				summary_df = pd.concat([summary_df, pd.DataFrame({'ne': [ne], 'scale': [scale], 'lower': [lower], 'upper' : [upper], 'prob': [prob], 'prob_minor': [prob_minor], 'prob_adj_minor': [prob_adj_minor]})])
	for i in range(len(bins) - 1):
		upper = bins[i+1]
		lower = bins[i]
		prob = len(df[(df['AF'] > lower) & (df['AF'] <= upper)]) / len(df)
		prob_minor = len(df[(df['AF'] > lower) & (df['AF'] <= min(upper, 0.5))]) / len(df[df['AF'] <= 0.5])
		prob_adj_minor = len(df[(df['AF_adj_minor'] > lower) & (df['AF_adj_minor'] <= upper)]) / len(df)
		summary_df = pd.concat([summary_df, pd.DataFrame({'ne': ["sample"], 'scale': ["sample"], 'lower': [lower], 'upper' : [upper], 'prob': [prob], 'prob_minor': [prob_minor], 'prob_adj_minor': [prob_adj_minor]})])
	
	summary_df.to_csv(f"sim_q_syn_{clip_gen_10k}/syn_summary{suffix}.txt", sep = "\t", index = False)

	sample = summary_df[summary_df['ne']=="sample"]['prob_minor'].to_numpy()
	for scale in scales:
		for ne in nes:
			sim = summary_df[(summary_df['ne']==ne) & (summary_df['scale']==scale)]['prob_minor'].to_numpy()
#			kl = sum(sim * np.log(sim / sample))
			kl = sum(sample * np.log(sample / sim))
			print(ne, scale, kl)
		

if __name__=="__main__":
	main(sys.argv[1], sys.argv[2])
