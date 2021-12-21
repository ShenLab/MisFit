import esm
import os
def gen_sequences(train):

    fasta_path='/data/alant/ESM/fasta_files/0.fasta'
 
    if not train:
        fasta_path='/data/alant/ESM/fasta_files/20000.fasta'
        embed_path='/data/alant/ESM/test_embeddings'
    for header,sequence in esm.data.read_fasta(fasta_path):
        header=header.split('-')
        target=header[0][1:]+'-'+header[1]+'.fasta'
        print (target)
        os.system('mv {} {}'.format(os.path.join('/data/alant/ESM/fasta_files/orig_sequences',target),os.path.join('/data/alant/ESM/fasta_files/seq')))



gen_sequences(train=False)
