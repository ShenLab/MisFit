import pandas as pd
import numpy as np
from Bio.Seq import Seq

def revcom(context):
	return str(Seq(context).reverse_complement())

def main():
	freq_dt = pd.read_table("context_count.txt")
	freq_dt.columns = ['context', 'count_1']
	freq_dt_rev = freq_dt.copy()
	freq_dt_rev['context'] = freq_dt_rev['context'].apply(revcom)
	freq_dt_rev.columns = ['context', 'count_2']
	freq_dt = pd.merge(freq_dt, freq_dt_rev)
	freq_dt['count'] = freq_dt['count_1'] + freq_dt['count_2']

	codon_dt = pd.read_table("codon_table.txt")
	codon_dt.columns = ['context', 'AA3', 'AA', 'name']
	codon_dt = codon_dt[codon_dt['AA']!="O"]

	AA_dt = pd.merge(freq_dt[['context', 'count']], codon_dt[['context', 'AA']])
	AA_dt = AA_dt.groupby(['AA'])['count'].sum().reset_index()
	total_count = AA_dt['count'].sum()
	AA_dt['frequency'] = AA_dt['count'] / total_count
	AA_dt = AA_dt.sort_values(by = "AA")
	AA_dt.to_csv("AA_frequency.txt", sep = "\t", index = False)
	AA_freq = AA_dt['frequency'].to_numpy()
	AA_freq = np.log(AA_freq)
	np.save("log_AA_freq.npy", AA_freq)

if __name__=="__main__":
	main()
