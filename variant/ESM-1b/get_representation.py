import torch
import numpy as np
from Bio import SeqIO
import gzip
import esm
from os.path import exists

cuda = torch.device('cuda:3') 

def main():

	model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
	batch_converter = alphabet.get_batch_converter()
	model = model.eval().to(cuda)

	# prepare data
	# sequence connot exceed length of 1024
	MAX_L = 1000
	SLIDING = 800
	OVERLAP = MAX_L - SLIDING
	data = []
	
	input_file = open("../list/geneset_uniprot_len.txt", "r")
	input_file.readline()
	for line in input_file:
		uniprot_id = line.split("\t")[0]
		length = int(line.split("\t")[-1])
		seq_filename = "../pep/uniprot_seq/" + uniprot_id + ".fasta.gz"
		with gzip.open(seq_filename, "rt") as f:
			record = SeqIO.read(f, "fasta")
		for part in range(length // SLIDING + 1):
			start = SLIDING * part
			end = start + MAX_L
			if end >= length:
				data.append((uniprot_id + "_" + str(part + 1), str(record.seq)[start:]))
				break
			else:
				data.append((uniprot_id + "_" + str(part + 1), str(record.seq)[start:end]))
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
		for i, label in enumerate(batch_labels):
			np.save("repr_" + str(MAX_L) + "_" + str(OVERLAP) + "/" + label + ".npy", repr_np[i, :, :])

		if (batch_index % 1000 == 0):
			print(str(batch_index) + " sequences processed.")

if __name__ == "__main__":
	main()
