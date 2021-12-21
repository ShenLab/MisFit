
import os


os.system('python extract.py esm1b_t33_650M_UR50S {} /data/alant/ESM/train_embeddings/orig \
 --include mean per_tok'.format(os.path.join('/data/alant/ESM/fasta_files','{}.fasta'.format('0_orig'))))
