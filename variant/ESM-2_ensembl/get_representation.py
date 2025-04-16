import torch
import numpy as np
from Bio import SeqIO
import gzip
import esm
from os.path import exists
from scipy.special import softmax

cuda = torch.device('cuda:1') 

def main():

	model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
	batch_converter = alphabet.get_batch_converter()
	model = model.eval().to(cuda)

	# prepare data
	# sequence connot exceed length of 1024
	MAX_L = 600
	SLIDING = 400
	OVERLAP = MAX_L - SLIDING
	data = []
	
	input_file = open("../list2/geneset_len.txt", "r")
	input_file.readline()
	for line in input_file:
		ensembl_id = line.split("\t")[0]
		length = int(line.split("\t")[-1])
		seq_filename = "../pep2/ensembl_seq/" + ensembl_id + ".fasta.gz"
		with gzip.open(seq_filename, "rt") as f:
			record = SeqIO.read(f, "fasta")
		for part in range(length // SLIDING + 1):
			start = SLIDING * part
			end = start + MAX_L
			if end >= length:
				data.append((ensembl_id + "_" + str(part + 1), str(record.seq)[start:]))
				break
			else:
				data.append((ensembl_id + "_" + str(part + 1), str(record.seq)[start:end]))
	input_file.close()

	batch_size = 100

	for batch_index in range(0, len(data), batch_size):
		batch_labels, batch_strs, batch_tokens = batch_converter(data[batch_index:(batch_index + batch_size)])
		batch_tokens = batch_tokens.to(cuda)

		# get per token representation
		with torch.no_grad():
			results = model(batch_tokens, repr_layers=[33])

		representation = results["representations"][33]
		# representation dimension: batch * (MAX_L + 2) * 1280
		repr_np = representation.cpu().numpy()

		logits = results["logits"]
		#  dimension: batch * (MAX_L + 2) * 33, while 33 is for 33 tokens
		repr_np = logits.cpu().numpy()
		ref = np.take_along_axis(repr_np, np.expand_dims(batch_tokens.cpu().numpy(), -1), -1)
		repr_np = np.take(repr_np, indices, axis = 2) # B * (MAX_L + 2) * 20
		repr_np = ref - repr_np
		for i, label in enumerate(batch_labels):
			np.save("logits_" + str(MAX_L) + "_" + str(OVERLAP) + "/" + label + ".npy", repr_np[i, 1:(batch_l[i]+1), :])
			np.save("repr_" + str(MAX_L) + "_" + str(OVERLAP) + "/" + label + ".npy", repr_np[i, :, :])

		if (batch_index % 1000 == 0):
			print(str(batch_index) + " sequences processed.")

if __name__ == "__main__":
	main()


