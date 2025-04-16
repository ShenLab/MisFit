import gzip
import re

def create_set(input_filename, header = True, column = 1):
    input_file = open(input_filename, "r")
    if header:
        line = input_file.readline()
    canonical_set = set() 
    for line in input_file:
        line_split = line.strip().split("\t")
        canonical_set.add(line_split[column])
    input_file.close()
    return canonical_set

def process_csq(csq, canonical_set):
    fields = csq.split("|")
    if fields[6] in canonical_set:
        # Feature field corresponding to transcript_id
        if fields[-4]=="HC":
            # Feature field conrresponding to LoF by LOFTEE
            return fields
    return None


def main():
    canonical_set = create_set("geneset_uniprot.txt", column = 2)
    input_filename = "all_pos_alt_loftee.vcf.gz"
    lof_filename = "all_pos_alt_HClof_sep_canonical.vcf"

    input_file = gzip.open(input_filename, "rt")
    lof_file = open(lof_filename, "w")

    count = 0

    for line in input_file:
        if re.search(r"^#", line):
            print(line.strip(), file = lof_file)
            continue
        if count % 1000000 == 0:
            print(str(count) + " variants processed.")
        count += 1
        line_split = line.strip().split("\t")
        csq_all = re.search(r"CSQ=(.*)$", line_split[7]).group(1).split(",")
        for csq in csq_all:
            all_fields = process_csq(csq, canonical_set)
            if all_fields is not None:
                print(*(line_split[0:7] + ["CSQ=" + csq]), sep = "\t", file = lof_file)

    input_file.close()
    lof_file.close()


if __name__ == "__main__":
    main()
