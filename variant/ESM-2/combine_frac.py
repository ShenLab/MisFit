import pandas as pd
import numpy as np

L = 1000
SLIDING = 800
OVERLAP = L - SLIDING
OVERLAP_TYPE = "weighted"
AA_list = "ACDEFGHIKLMNPQRSTVWY"

def _gen_overlap_weight(start, end, mask_start, mask_end):
    full_weights = np.ones(shape = (L, 1)) 
    full_weights[(end - start + 1):, :] = 0.
    if OVERLAP_TYPE == "weighted":
        weights = np.array([(i+1)/(OVERLAP+1) for i in range(OVERLAP)])
        weights = np.reshape(weights, (OVERLAP, 1)) 
        if mask_start > start:
            full_weights[0:OVERLAP] = weights
        if mask_end < end:
            full_weights[(L-OVERLAP):L] = weights[::-1,:]
    else:
        full_weights[:(mask_start - start), :] = 0.
        full_weights[(mask_end - start + 1):, :] = 0.
    return full_weights[:(end - start + 1),:].astype(np.float32)

def main():
	segset = pd.read_table(f"seg_list_{L}_{OVERLAP}.txt")
	genes = segset['UniprotID'].unique()
	for gene in genes:
		subset = segset[segset['UniprotID']==gene]
		l = subset['end'].max()
		full_score = np.zeros((l, len(AA_list)))
		for _, row in subset.iterrows():
			frac = row['frac']
			start = row['start']
			end = row['end']
			mask_start = row['unmask_start']
			mask_end = row['unmask_end']
			weight = _gen_overlap_weight(start, end, mask_start, mask_end)
			frac_score = np.load(f"logits_{L}_{OVERLAP}/{gene}_{frac}.npy")
			frac_score *= weight
			full_score[(start-1):end] += frac_score # (l, A)
		result = {
			"Protein_position": [i // len(AA_list) + 1 for i in range(len(AA_list) * l)], # zero based
            "AA_alt": list(AA_list) * l,
            "ESM": np.reshape(full_score, (-1)), 
        }
		pd.DataFrame(result).to_csv(f"logits_merged/{gene}.txt.gz", sep = "\t", index = False)

	

if __name__=="__main__":
	main()
