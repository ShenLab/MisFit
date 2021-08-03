
import os
#l=[0,20000,40000,60000,80000,100000]
#for i in l:
os.system('python extract.py esm1b_t33_650M_UR50S {} /data/alant/ESM/train_embeddings/orig \
 --include mean per_tok'.format(os.path.join('/data/alant/ESM/fasta_files','{}.fasta'.format('0_orig'))))
