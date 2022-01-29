import pyfastx
from pathlib import Path

def main():
    seqfile = Path.home().joinpath("data/annotation/.vep/homo_sapiens/104_GRCh38/Homo_sapiens.GRCh38.dna.toplevel.fa.gz")
    seqs = pyfastx.Fasta(str(seqfile))
    print("sequence successfully read!")
    cds_filename = "all_pos.sorted.txt"
    cds_file = open(cds_filename, "r")
    pos_filename = "all_pos_alt.txt"
    pos_file = open(pos_filename, "w")
    log_filename = "all_pos_alt.log"
    log_file = open(log_filename, "w")

    chrom = "0"
    start = 0
    end = 0

    line = cds_file.readline()
    count = 0
    while line:
        count += 1
        if count % 10000 == 0:
            print(f"processed {count} CDS", file = log_file)
        position = line.strip().split("\t")
        chrom_new = position[0]
        start_new = int(position[1])
        end_new = int(position[2])

        if (chrom_new==chrom) & (start_new<=end):
            print(f"overlap: {chrom}: {start}-{end} and {start_new}-{end_new}", file = log_file)
            if (end_new<=end):
                line = cds_file.readline()
                continue
            else:
                start = end + 1
        else:
            start = start_new
        chrom = chrom_new
        end = end_new

        region = seqs[chrom][(start-2):(end+1)].seq
        for pos in range(end - start + 1):
            context = region[(pos):(pos+3)]
            ref = context[1]
            if ref in ['A', 'C', 'G', 'T']:
                for alt in ['A', 'C', 'G', 'T']:
                    if ref==alt:
                        continue
                    print(f"{chrom}\t{start+pos}\t{ref}\t{alt}\t{context}", file = pos_file)
        line = cds_file.readline()

    cds_file.close()
    pos_file.close()
    log_file.close()

if __name__ == "__main__":
    main()

