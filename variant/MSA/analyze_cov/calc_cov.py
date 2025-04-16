import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial import distance_matrix
from Bio.Align import substitution_matrices
matrix = substitution_matrices.load('BLOSUM62')
MAX_L = 1000
SLIDING = 800
OVERLAP = MAX_L - SLIDING
PSUDO = 0.1
MIN_MUTRATE = 1e-7
MAX_SCORE = -1
#MIN_MUTRATE = 0
#MAX_SCORE = 0

def select_variants(df, frac):
	start = (frac - 1) * SLIDING
	df2 = df.loc[(df['uniprot_pep_pos'] > start) & (df['uniprot_pep_pos'] <= start + MAX_L)].copy()
	df2 = df2[df2['gnomad_mu'] > MIN_MUTRATE]
	df2['s'] = df2['gnomad_mu'] / (df2['UKBB_AC'] + PSUDO) * (df2['UKBB_AN'] + PSUDO)
	df2['s_rank'] = df2['s'].rank(ascending = True, pct = True)
	if len(df2) < 2:
		return None, None
	df2['score'] = df2.apply(lambda x: matrix[x['ref_aa']][x['alt_aa']], axis = 1)
	df2 = df2[df2['score'] <= MAX_SCORE]
	df2 = df2.sort_values('gnomad_mu', ascending = False).sort_values('score', ascending=True).drop_duplicates(['uniprot_pep_pos'])
	df2 = df2.sort_values('uniprot_pep_pos', ascending = True)
	if len(df2) > 0:
		positions = df2['uniprot_pep_pos'].to_numpy() - start - 1
		s = df2['s_rank'].to_numpy()
		s = np.reshape(s, (-1, 1))
		return positions, s
	else:
		return None, None
	
def compare_cov(cov_norm, positions, s):
	cov_norm = np.take(cov_norm, positions, axis = 0)
	cov_norm = np.take(cov_norm, positions, axis = 1)
	dist_s = distance_matrix(s, s, p = 1)
	mask = np.ones((cov_norm.shape[0], cov_norm.shape[1]))
	mask = mask - np.tril(mask)
	mask = mask.astype('bool')
	mask = np.reshape(mask, (-1))
	cov_norm = np.reshape(cov_norm, (-1))[mask]
	dist_s = np.reshape(dist_s, (-1))[mask]
	return spearmanr(cov_norm, dist_s).correlation

def process_protein(uniprot_id, frac):
	try:
		df = pd.read_table("../../mis_info/mis_info_uniprot/" + uniprot_id + "_uniaa.txt.gz")
		cov_norm = np.load("../coev_norm_" + str(MAX_L) + "_" + str(OVERLAP) + "/" + uniprot_id + "_" + str(frac) + ".npy")
	except:
		return None, None
	positions, s  = select_variants(df, frac)
	if positions is None:
		return None, None
	r = compare_cov(cov_norm, positions, s)
	return r, len(positions)

def main():
	all_genes = pd.read_table("../../list/geneset_uniprot_len.txt")
	summary = open("coev_norm_s_corr.txt", "w", 1)
	print("UniprotID\tTranscriptID\tGeneID\tFrac\tNum_pos\tCor", file = summary)
	for i, row in all_genes.iterrows():
		uniprot_id = row['UniprotID']
		transcript_id = row['TranscriptID']
		gene_id = row['GeneID']
		length = row['Length']
		for frac in range(1, max(2, (length - MAX_L - 1) // SLIDING + 3)):
			r, num = process_protein(uniprot_id, frac)
			if (num is not None) and (num > 1):
				print(f"{uniprot_id}\t{transcript_id}\t{gene_id}\t{frac}\t{num}\t{r}", file = summary)
		if i % 2000 == 0:
			print(f"processed {i} proteins")
	summary.close()

if __name__=="__main__":
	main()


