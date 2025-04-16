import numpy as np
import os.path

MAX_L = 1000
OVERLAP = 200
SCALE = 0.5

def decompose(corr_matrix, scale = 1., clip_value = 0.):
	corr_matrix *= scale
	np.fill_diagonal(corr_matrix, 1.)
	value, vector = np.linalg.eig(corr_matrix)
	clipped_value = np.clip(value, clip_value, None)
	U = np.sqrt(clipped_value) * vector
	U = U / np.sqrt(np.sum(U*U, -1, keepdims = True))
	return U

def gen_contact_decomp(filename, scale = SCALE, clip_value = 1e-6):
	contact = np.zeros(shape = (MAX_L, MAX_L))
	data = np.load(filename)
	dim = data.shape[-1]
	contact[0:dim,0:dim] = data
	U = decompose(contact, scale = scale, clip_value = clip_value)
	return U

def main():
	input_dir = f"contact_{MAX_L}_{OVERLAP}"
	output_dir = f"contact_decomp_{MAX_L}_{OVERLAP}_{SCALE}"
	for filename in os.listdir(input_dir):
		U = gen_contact_decomp(f"{input_dir}/{filename}")
		np.save(f"{output_dir}/{filename}", U)

if __name__=="__main__":
	main()
