chroms = [str(i) for i in range(1,23)] + ["X", "Y"]
for i in chroms:
	chrom = "chr" + i
	print(f"sbatch -J {chrom} annot_genome.sh {chrom}")
