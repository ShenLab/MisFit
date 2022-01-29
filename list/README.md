
`get_geneset.R`
Get protein-coding gene and corresponding transcript. Transcript is the MANE-select transcript of the gene, or ensembl canonical if no MANE.
input: 
`ensembl_104_transcripts.txt` downloaded from BioMart of Ensembl 104, protein-coding genes with HGNC ID only
output: 
`geneset.txt` (first line as header) 19349 genes in total (13 MT, 837 X, 46 Y, 18453 auto). Ensembl gene ID unique. 
PINX1, POLR2J3, SIGLEC5, TBCE have multiple ensembl gene IDs. 


`get_cds_pos.py`
Get all SNVs from CDS(+-2) in annotation file.
input: 
`gencode.v38.basic.annotation.gff3.gz` downloaded from Gencode v38 (corresponding to Ensembl 104)
`geneset.txt`
output:
`all_pos.txt` (chrom start end)

`all_pos.sorted.txt` is `all_pos.txt` sorted by position
command:
```
sort -k1,1 -k2,2n -k3,3n all_pos.txt > all_pos.sorted.txt
```

`get_ref_context.py`
Get genomic context (for reach the mutation rate) and all SNVs.
input:
`all_pos.sorted.txt`
output:
`all_pos_alt.txt` ~100M variants
`all_pos_alt.log` overlapped regions
