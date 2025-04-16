from Bio.Seq import Seq
import re

def is_cpg(context, ref, alt):
	assert context[1] == ref
	if (ref == "C") and (context[2] == "G") and (alt == "T"):
		return True
	elif (ref == "G") and (context[0] == "C") and (alt == "A"):
		return True
	return False


class mutation_dict():
	def __init__(self):
		self.dict = {}

	def add(self, context, ref, alt, methylation_level, mu):
		assert context[1] == ref
		if not (context in self.dict):
			self.dict[context] = {}
		if is_cpg(context, ref, alt):
			if not (alt in self.dict[context]):
				self.dict[context][alt] = {}
			self.dict[context][alt][methylation_level] = mu
		else:
			self.dict[context][alt] = mu

	def add_both(self, context, ref, alt, methylation_level, mu):
		self.add(context, ref, alt, methylation_level, mu)
		self.add(str(Seq(context).reverse_complement()), str(Seq(ref).reverse_complement()), str(Seq(alt).reverse_complement()),
			methylation_level, mu)

	def get_mu(self, context, ref, alt, methylation_level = "0"):
		assert context[1] == ref
		if not (context in self.dict):
			return None
		if is_cpg(context, ref, alt):
			return self.dict[context][alt][methylation_level]
		else:
			return self.dict[context][alt]


def main():
	input_filename = "all_pos_alt_context_meth.vcf"
	output_filename = "all_pos_alt_gnomad_mu.vcf"

	mutrate_file = open("mutation_rate.tsv", "r")
	line = mutrate_file.readline()
	rates = mutation_dict()

	for line in mutrate_file:
		line_split = line.strip().split("\t")
		context = line_split[0]
		ref = line_split[1]
		alt = line_split[2]
		methylation_level = line_split[3]
		mu = line_split[4]
		rates.add_both(context, ref, alt, methylation_level, mu)

	mutrate_file.close()

	input_file = open(input_filename, "r")
	output_file = open(output_filename, "w")
	
	for _ in range(2):
		line = input_file.readline()
		print(line, end = "", file = output_file)
	print("##INFO=<ID=gnomAD_mu,Number=1,Type=String,Description=\"gnomAD mutation rate\">", file = output_file)
	print("##INFO=<ID=CpG,Number=0,Type=Flag,Description=\"C to T variant at CpG site\">", file = output_file)
	
	count = 0
	for line in input_file:
		if count % 1000000 == 0:
			print(f"{count} variants processed.")
		count += 1
		if re.search(r"^#", line):
			print(line, end = "", file = output_file)
		else:
			line_split = line.strip().split("\t")
			ref = line_split[3]
			alt = line_split[4]
			context = re.search(r"CONTEXT=(\w+)", line_split[7]).group(1)
			meth_search = re.search(r"METH=(\w+)", line_split[7])
			if meth_search:
				methylation_level = meth_search.group(1)
			else:
				methylation_level = "0"
			CpG = is_cpg(context, ref, alt)
			mutation_rate = rates.get_mu(context, ref, alt, methylation_level)

			if CpG:
				line = line.strip() + ";CpG"
			if mutation_rate is not None:
				line = line.strip() + ";gnomAD_mu=" + mutation_rate

			print(line.strip(), file = output_file)
			

	input_file.close()
	output_file.close()

if __name__ == '__main__':
	main()
