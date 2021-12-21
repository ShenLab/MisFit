
from Bio import SeqIO
import csv
import os
from Bio import Seq


def crop(target_len,include,string):
    if (len(string)<=target_len):
        return string
    
    if (include-target_len//2>=0 and len(string)>include+target_len//2):
        return (string[include-target_len//2:include+target_len//2])
    if (include<target_len):
        return (string[:target_len])
    else:
    
        return (string[len(string)-target_len:])
def data(path,write_path):    
    protein_data=[]
    with open(path) as csv_file:
            replacement_index=0
            replacement_aa_index=0
            id_index=0
            ref_aa=0
            target=0
            csv_read = csv.reader(csv_file, delimiter='\t')
            file_root='/data/hz2529/zion/MVPContext/feature'
            line_count = 0
           
            for row in csv_read:
                if (line_count==0):
                    replacement_index=row.index("aa_pos") #integer
                    replacement_aa_index=row.index('alt_aa') #string
                    id_index=row.index('transcript_id')
                    ref_index=row.index('ref_aa')
                    target=row.index('target')
                    line_count += 1

                else:
                     
                    protein_id=row[id_index]
                     
                    seq_path=os.path.join(file_root,protein_id+'.fasta')
                    try:
                        open(seq_path)
                    except FileNotFoundError:
                        print ("{} was not found".format(protein_id))
                        continue
                    for sequence in SeqIO.parse(open(seq_path), 'fasta'):
                        header='{}-{}-{}'.format(line_count-1,sequence.id,row[target])
                       # for i in sequence:
                       #     protein_seq_path[protein_id]=str(i.seq)
                        sequence.id=header
                        sequence.description=header
                        protein_sequence=str(sequence.seq)
                        if( protein_sequence[int(row[replacement_index])-1]!=row[ref_index]):
                             raise Exception("Reference protein does not match")
                        protein_sequence[int(row[replacement_index])-1]=row[replacement_aa_index] #replace aa with missense var.
                  #       
                        protein_sequence = ''.join(protein_sequence)
                        protein_sequence=crop(256,int(row[replacement_index])-1,protein_sequence)
                        sequence.seq=Seq.Seq(protein_sequence)
                        if line_count==1:
                            print ('Writing...')
                        line_count+=1
                        with open(os.path.join(write_path,'{}.fasta'.format(header)),'x') as handle1:
                            try:
                                 SeqIO.write(sequence,handle1,'fasta')
                            except FileNotFoundError:
                                 print ("{} not found".format(sequence))


if __name__=='__main__':

    data(path='/data/hz2529/neo/repos/genetic_variants_data/train/2021_v20/0.csv',write_path='/data/alant/ESM/fasta_files/orig_sequences')
