import re
import gzip

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
        return fields
    else:
        return None

def get_type(consequence):
    if re.search("stop_gained|stop_lost|start_lost|frameshift|splice_acceptor|splice_donor", consequence):
        return 2
    if re.search("missense", consequence):
        return 0
    if consequence=="synonymous_variant":
        return 1
    return None


def main():
    canonical_set = create_set("geneset_uniprot.txt", column = 2)
    input_filename = "all_pos_alt_rawcsq.vcf.gz"
    prefix = "all_pos_alt"
    missense_filename = prefix + "_missense_sep_canonical.vcf"
    synonymous_filename = prefix + "_synonymous_sep_canonical.vcf"
    #lof_filename = prefix + "_lof_sep_canonical.vcf"

    input_file = gzip.open(input_filename, "rt")
    missense_file = open(missense_filename, "w")
    synonymous_file = open(synonymous_filename, "w")
    #lof_file = open(lof_filename, "w")

    count = 0

    for line in input_file:
        if re.search(r"^#", line):
            print(line.strip(), file = missense_file)
            print(line.strip(), file = synonymous_file)
            #print(line.strip(), file = lof_file)
            continue
        if count % 1000000 == 0:
            print(str(count) + " variants processed.")
        count += 1
        line_split = line.strip().split("\t")
        csq_all = re.search(r"CSQ=(.*)$", line_split[7]).group(1).split(",")
        for csq in csq_all:
            all_fields = process_csq(csq, canonical_set)
            if all_fields is not None:
                csq_type = get_type(all_fields[1])
                if (csq_type==0): # missense
                    print(*(line_split[0:7] + ["CSQ=" + csq]), sep = "\t", file = missense_file)
                elif (csq_type==1): # synonymous
                    print(*(line_split[0:7] + ["CSQ=" + csq]), sep = "\t", file = synonymous_file)
                #elif (csq_type==2): # lof
                    #print(*(line_split[0:7] + ["CSQ=" + csq]), sep = "\t", file = lof_file)

    input_file.close()
    missense_file.close()
    synonymous_file.close()
    #lof_file.close()


if __name__ == "__main__":
    main()
