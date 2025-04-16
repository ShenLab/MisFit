import pandas as pd
import json

def segment(length, window, overlap, processed_window, processed_sliding, processed_max_length, adjust_last):
	if adjust_last:
		assert processed_window - processed_sliding >= window
	else:
		assert (window - overlap) % processed_sliding == 0
	if length <= window:
		frac = [1]
		struct_frac = [1]
		start = [1]
		end = [length]
		struct_start = [1]
		struct_end = [length]
	else:
		frac = []
		struct_frac = []
		start = []
		end = []
		struct_start = []
		struct_end = []
		for curr_frac in range(1, (length - window - 1) // (window - overlap) + 3):
			curr_start = (window - overlap) * (curr_frac - 1) + 1
			curr_end = curr_start + window - 1
			if curr_end > length:
				curr_end = length
				if adjust_last:
					curr_start = curr_end - window + 1
			if length <= processed_max_length:
				curr_struct_frac = 1
			else:
				curr_struct_frac = max(int(round(((curr_start + curr_end) - (1 + processed_window)) / 2 / processed_sliding, 0)), 0) + 1
				curr_struct_frac = min(curr_struct_frac, (length - processed_window - 1) // (processed_sliding) + 2)
			curr_struct_start = curr_start - (curr_struct_frac - 1) * processed_sliding
			curr_struct_end = curr_end - curr_start + curr_struct_start
			frac.append(curr_frac)
			struct_frac.append(curr_struct_frac)
			start.append(curr_start)
			end.append(curr_end)
			struct_start.append(curr_struct_start)
			struct_end.append(curr_struct_end)
	return frac, start, end, struct_frac, struct_start, struct_end
# 1-based, including start and end

def main():
	setting_file = open("setting.json", "r")
	setting = json.load(setting_file)
	setting_file.close()
	geneset = pd.read_table("geneset_uniprot_len.txt")
	output_file = open(f"{setting['protein_list']}_{setting['segment_length']}_{setting['overlap_length']}.txt", "w")
	window = setting['struc_window']
	sliding = setting['struc_sliding']
	max_l = setting['struc_max_l']
	print("UniprotID", "TranscriptID", "GeneID", "Symbol", "Chrom", "frac", "start", "end", "struc_frac", "struc_start", "struc_end", "unmask_start", "unmask_end", sep = "\t", file = output_file)
	for _, row in geneset.iterrows():
		frac, start, end, struct_frac, struct_start, struct_end = segment(row['Length'], setting['segment_length'], setting['overlap_length'], window, sliding, max_l, setting['adjust_last'])
		unmask_start = [pos for pos in start]
		unmask_end = [pos for pos in end]
		for i in range(len(frac)):
			if i < len(frac) - 1:
				if start[i+1]<= end[i]:
					unmask_end[i] = (start[i+1] + end[i]) // 2
					unmask_start[i+1] = unmask_end[i] + 1
			print(row['UniprotID'], row['TranscriptID'], row['GeneID'], row['Symbol'], row['Chrom'], frac[i], start[i], end[i], struct_frac[i], struct_start[i], struct_end[i], unmask_start[i], unmask_end[i], sep = '\t', file = output_file)
	output_file.close()

if __name__=="__main__":
	main()




