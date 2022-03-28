# get the top20 result
import sys

fname = sys.argv[1] # input file
freq_pairs = {}
topK = 20

with open(fname) as f:
	for line in f.readlines():
		pair, count = line.strip().split('\t')
		freq_pairs[pair] = int(count)

if len(freq_pairs) > topK:
	result = sorted(freq_pairs.items(), key = lambda x:x[1], reverse = True)[:topK]
	# sort according to value
else:
	result = sorted(freq_pairs.items(), key = lambda x:x[1], reverse = True)

for pair in result:
	print(pair)
