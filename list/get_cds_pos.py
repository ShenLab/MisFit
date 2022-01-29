import re
import gzip
import pandas as pd

def process_gff(line):
    line_split = line.strip().split("\t")
    chrom = line_split[0]
    feature = line_split[2]
    start = line_split[3]
    end = line_split[4]
    info = line_split[8]

    if (feature == "CDS") or (feature == "transcript"):
        transcript_search = re.search(r"transcript_id=(\w+)", info)
        if transcript_search:
            transcript_id = transcript_search.group(1)
        else:
            transcript_id = None
    else:
        transcript_id = None
    return feature, transcript_id, chrom, start, end

def rename_chrom(chrom):
    chr_search = re.search(r"^chr(\w+)", chrom)
    if chr_search:
        new_chrom = chr_search.group(1)
        if new_chrom == "M":
            return "MT"
        else:
            return new_chrom
    return chrom

def main():
    # inputs
    annot_filename = "gencode.v38.basic.annotation.gff3.gz"
    list_filename = "geneset.txt"

    annot_file = gzip.open(annot_filename, "rt")

    # outputs
    pos_filename = "all_pos.txt"
    pos_file = open(pos_filename, "w")

    # get gene lists
    list_df = pd.read_table(list_filename, sep = "\t", names = ['gene', 'transcript', 'protein', 'name', 'strand', 'seq'])
    transcript_list = set(list_df['transcript'])

    # split line
    count = 0
    for line in annot_file:
        if re.search(r"^#", line):
            continue
        feature, transcript_id, chrom, start, end = process_gff(line)
        if transcript_id in transcript_list:
            if feature=="transcript":
                count += 1
                if count%1000 == 0:
                    print(f"processed {count} protein_coding genes")
            if feature=="CDS":
                # chromosome name as 1:22, X, Y, MT
                chrom = rename_chrom(chrom)
                # to allow capturing of splice_donor and splice_acceptor
                start = int(start) - 2
                end = int(end) + 2
                print(f"{chrom}\t{start}\t{end}", file = pos_file)

    annot_file.close()
    pos_file.close()

if __name__ == "__main__":
    main()
