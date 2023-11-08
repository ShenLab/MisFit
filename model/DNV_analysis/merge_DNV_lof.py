import pandas as pd
import os, sys
import json
from scipy.stats import spearmanr
import json

def main(prefix_dir, filename):
	DNV_df = pd.read_table(os.path.expanduser(f"~/data/model/DNV_analysis/{filename}.txt.gz"), dtype = {'Symbol': 'string'})

#	if not os.path.exists(os.path.expanduser(f"{prefix_dir}/DNV/")):
#		os.makedirs(os.path.expanduser(f"{prefix_dir}/DNV/"))

	function_df = pd.read_table(os.path.expanduser(f"{prefix_dir}/geneset_lof_s_combined.txt"), dtype = {'Symbol': 'string'})
	cols = [col for col in function_df.columns if col not in DNV_df.columns] + ['Symbol', 'TranscriptID']
	DNV_df = pd.merge(DNV_df, function_df[cols], how = "left")

	DNV_df.to_csv(os.path.expanduser(f"{prefix_dir}/{filename}_merged.txt.gz"), sep = "\t", index = False)


if __name__=="__main__":
	main(sys.argv[1], "ASD_DNV_lof_hg19")
	main(sys.argv[1], "NDD_DNV_lof_hg19")
#	main(sys.argv[1], "EA_DNV_lof_hg19")
#	main(sys.argv[1], "CDH_DNV_lof")
	main(sys.argv[1], "ASD_SPARK0_HClof")
