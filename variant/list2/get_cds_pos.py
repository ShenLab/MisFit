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
        transcript_search = re.search(r"transcript:(\w+)", info)
        if transcript_search:
            transcript_id = transcript_search.group(1)
        else:
            transcript_id = None
    else:
        transcript_id = None
    return feature, transcript_id, chrom, start, end


def main():
    # inputs
    annot_filename = "Homo_sapiens.GRCh38.112.chr.gff3.gz"
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
                # to allow capturing of splice_donor and splice_acceptor
                start = int(start) - 2
                end = int(end) + 2
                print(f"{chrom}\t{start}\t{end}", file = pos_file)

    annot_file.close()
    pos_file.close()

if __name__ == "__main__":
    main()
