import pandas as pd
import gzip

def main():
	filename = "gencode.v38.basic.annotation.gff3.gz"
	df = pd.read_table(filename, skiprows = 7, header = None)
	df = df[[0, 2, 3, 4]]
	df.columns = ["Chrom", "Feature", "Start", "End"]
	df = df[df["Feature"]=="exon"]
	for i in [str(i) for i in range(1, 23)] + ["X"]:
		chrom = "chr" + i
		df_sub = df[df['Chrom']==chrom].sort_values(by = ["Start", "End"])
		curr_end = 50000
		for _, row in df.iterrows():
			if (row['Start'] > curr_end + 1000) and (row['Start'] < curr_end + 10000):
				print(i, int(curr_end)+150, int(row['Start'])-150, sep = "\t")
			curr_end = max(row['End'], curr_end)


if __name__=="__main__":
	main()
