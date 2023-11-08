import pandas as pd
import sys
from sklearn.metrics import matthews_corrcoef

def main(prefix_dir):
	thresh_df = pd.read_table(f"{prefix_dir}/thresh_combined_summary.txt")
	df = pd.read_table(f"{prefix_dir}/DMS_combined_label.txt.gz")
	score_names = ["model_damage", "model_selection", "ESM-1b", "ESM-2", "AlphaMissense", "gMVP", "EVE", "CADD_phred", "REVEL", "PrimateAI", "MPC"]
	mcc_df = thresh_df[thresh_df['UniprotID']!="all"][['UniprotID','Symbol']]
	for score_name in score_names:
		mcc_df[score_name] = pd.NA
		thresh = thresh_df[thresh_df['UniprotID']=='all'][score_name].values[0]
		for i, row in mcc_df.iterrows():
			DMS_df = df[df['UniprotID']==row['UniprotID']][[score_name, 'target']].dropna()
			if len(DMS_df)==0:
				continue
			mcc = matthews_corrcoef(DMS_df['target'], DMS_df[score_name]>thresh)
			mcc_df.at[i, score_name] = mcc
	mcc_df.to_csv(f"{prefix_dir}/MCC_summary.txt", index = False, sep = "\t")

if __name__=="__main__":
	main(sys.argv[1])
