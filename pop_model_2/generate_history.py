import pandas as pd

def get_history(final_ne = 1e6, warmup = 0, clip = None):
	# Durbin 2014
	pop_df = pd.read_csv("EUR_pop.csv")
	Ne_discrete = pop_df['Ne'].values.tolist()
	T_discrete = pop_df['generation'].values.tolist()
	Ne_discrete.append(final_ne)
	T_discrete.append(0)
	Ne = [Ne_discrete[0]] * warmup * Ne_discrete[0]
	for i in range(len(T_discrete) - 1):
		Ne.append(Ne_discrete[i])
		gr = (Ne_discrete[i+1] / Ne_discrete[i]) ** (1 / (T_discrete[i] - T_discrete[i+1]))
		for t in range(T_discrete[i] - T_discrete[i+1] - 1):
			Ne.append(Ne[-1] * gr)
	Ne.append(final_ne)
	Ne_history = [int(n) for n in Ne]
	if clip is None:
		return Ne_history
	else:
		return Ne_history[-clip:]
