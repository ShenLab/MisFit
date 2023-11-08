import pandas as pd
import sys
from sklearn.metrics import matthews_corrcoef

def main(prefix_dir):
	thresh_df = pd.read_table(f"{prefix_dir}/thresh_combined_summary.txt")
	df = pd.read_table(f"{prefix_dir}/DMS_combined_label.txt.gz")
	score_names = ["model_damage", "model_selection", "ESM-1b", "ESM-2", "AlphaMissense", "gMVP", "EVE", "CADD_phred", "REVEL", "PrimateAI", "MPC"]
	f = open(f"{prefix_dir}/sensitivity_summary.txt", "w")
	for score_name in score_names:
		thresh = thresh_df[thresh_df['UniprotID']=='all'][score_name].values[0]
		sens = len(df[(df['target']==1) & (df[score_name] > thresh)]) / len(df[df['target']==1])
		print(score_name, sens, sep = "\t", file = f)
	f.close()

if __name__=="__main__":
	main(sys.argv[1])
